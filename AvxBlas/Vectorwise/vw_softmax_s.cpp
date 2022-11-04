#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_max_s.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

#include <math.h>

using namespace System;

#pragma unmanaged

__forceinline __m256 _mm256_softmaxexp_ps(__m256 x, __m256 x_max) {
    __m256 y = _mm256_sub_ps(x, x_max);
    __m256 z = _mm256_exp_ps(y);

    return z;
}

__forceinline __m256 _mm256_maxwise2_ps(__m256 x) {
    __m256 y = _mm256_max_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));

    return y;
}

__forceinline __m256 _mm256_sumwise2_ps(__m256 x) {
    __m256 y = _mm256_add_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));

    return y;
}

__forceinline __m256 _mm256_maxwise3_ps(__m256 x) {
    const __m256i _perm0 = _mm256_setr_epi32(2, 0, 1, 5, 3, 4, 6, 7);
    const __m256i _perm1 = _mm256_setr_epi32(1, 2, 0, 4, 5, 3, 6, 7);

    __m256 y = _mm256_max_ps(_mm256_max_ps(x, _mm256_permutevar8x32_ps(x, _perm0)), _mm256_permutevar8x32_ps(x, _perm1));

    return y;
}

__forceinline __m256 _mm256_sumwise3_ps(__m256 x) {
    const __m256i _perm0 = _mm256_setr_epi32(2, 0, 1, 5, 3, 4, 6, 7);
    const __m256i _perm1 = _mm256_setr_epi32(1, 2, 0, 4, 5, 3, 6, 7);

    __m256 y = _mm256_add_ps(_mm256_add_ps(x, _mm256_permutevar8x32_ps(x, _perm0)), _mm256_permutevar8x32_ps(x, _perm1));

    return y;
}

__forceinline __m256 _mm256_maxwise4_ps(__m256 x) {
    __m256 y = _mm256_max_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_max_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256 _mm256_sumwise4_ps(__m256 x) {
    __m256 y = _mm256_add_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_add_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256 _mm256_maxwise8_ps(__m256 x) {
    __m256 y = _mm256_max_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_max_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));
    __m256 w = _mm256_max_ps(z, _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(z), _MM_PERM_BADC)));

    return w;
}

__forceinline __m256 _mm256_sumwise8_ps(__m256 x) {
    __m256 y = _mm256_add_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_add_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));
    __m256 w = _mm256_add_ps(z, _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(z), _MM_PERM_BADC)));

    return w;
}

__forceinline __m256 _mm256_normal_asone_ps(__m256 x) {
    __m256 y = _mm256_and_ps(_mm256_set1_ps(NAN), _mm256_cmp_ps(x, x, _CMP_NEQ_UQ));
    __m256 z = _mm256_add_ps(_mm256_set1_ps(1), y);

    return z;
}

int vw_softmax_stride1_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride != 1 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 fills = _mm256_set1_ps(1);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        y0 = _mm256_normal_asone_ps(x0);
        y1 = _mm256_normal_asone_ps(x1);
        y2 = _mm256_normal_asone_ps(x2);
        y3 = _mm256_normal_asone_ps(x3);

        _mm256_stream_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        y0 = _mm256_normal_asone_ps(x0);
        y1 = _mm256_normal_asone_ps(x1);

        _mm256_stream_x2_ps(y_ptr, y0, y1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);

        y0 = _mm256_normal_asone_ps(x0);

        _mm256_stream_x1_ps(y_ptr, y0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        y0 = _mm256_normal_asone_ps(x0);

        _mm256_maskstore_ps(y_ptr, mask, y0);
    }

    return SUCCESS;
}

int vw_softmax_stride2_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride != 2 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0_max = _mm256_maxwise2_ps(x0);
        x1_max = _mm256_maxwise2_ps(x1);
        x2_max = _mm256_maxwise2_ps(x2);
        x3_max = _mm256_maxwise2_ps(x3);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);
        y2 = _mm256_softmaxexp_ps(x2, x2_max);
        y3 = _mm256_softmaxexp_ps(x3, x3_max);

        y0_sum = _mm256_sumwise2_ps(y0);
        y1_sum = _mm256_sumwise2_ps(y1);
        y2_sum = _mm256_sumwise2_ps(y2);
        y3_sum = _mm256_sumwise2_ps(y3);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);
        z2 = _mm256_div_ps(y2, y2_sum);
        z3 = _mm256_div_ps(y3, y3_sum);

        _mm256_stream_x4_ps(y_ptr, z0, z1, z2, z3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x0_max = _mm256_maxwise2_ps(x0);
        x1_max = _mm256_maxwise2_ps(x1);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);

        y0_sum = _mm256_sumwise2_ps(y0);
        y1_sum = _mm256_sumwise2_ps(y1);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);

        _mm256_stream_x2_ps(y_ptr, z0, z1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x1_ps(x_ptr, x0);

        x0_max = _mm256_maxwise2_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise2_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_stream_x1_ps(y_ptr, z0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r * stride);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0_max = _mm256_maxwise2_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise2_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_maskstore_ps(y_ptr, mask, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride3_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride != 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask6 = _mm256_setmask_ps(6);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE) {
        x0 = _mm256_maskload_ps(x_ptr, mask6);
        x1 = _mm256_maskload_ps(x_ptr + 6, mask6);
        x2 = _mm256_maskload_ps(x_ptr + 12, mask6);
        x3 = _mm256_maskload_ps(x_ptr + 18, mask6);

        x0_max = _mm256_maxwise3_ps(x0);
        x1_max = _mm256_maxwise3_ps(x1);
        x2_max = _mm256_maxwise3_ps(x2);
        x3_max = _mm256_maxwise3_ps(x3);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);
        y2 = _mm256_softmaxexp_ps(x2, x2_max);
        y3 = _mm256_softmaxexp_ps(x3, x3_max);

        y0_sum = _mm256_sumwise3_ps(y0);
        y1_sum = _mm256_sumwise3_ps(y1);
        y2_sum = _mm256_sumwise3_ps(y2);
        y3_sum = _mm256_sumwise3_ps(y3);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);
        z2 = _mm256_div_ps(y2, y2_sum);
        z3 = _mm256_div_ps(y3, y3_sum);

        _mm256_storeu_ps(y_ptr, z0);
        _mm256_storeu_ps(y_ptr + 6, z1);
        _mm256_storeu_ps(y_ptr + 12, z2);
        _mm256_maskstore_ps(y_ptr + 18, mask6, z3);

        x_ptr += 24;
        y_ptr += 24;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        x0 = _mm256_maskload_ps(x_ptr, mask6);
        x1 = _mm256_maskload_ps(x_ptr + 6, mask6);

        x0_max = _mm256_maxwise3_ps(x0);
        x1_max = _mm256_maxwise3_ps(x1);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);

        y0_sum = _mm256_sumwise3_ps(y0);
        y1_sum = _mm256_sumwise3_ps(y1);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);

        _mm256_storeu_ps(y_ptr, z0);
        _mm256_maskstore_ps(y_ptr + 6, mask6, z1);

        x_ptr += 12;
        y_ptr += 12;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r >= AVX2_FLOAT_STRIDE / 4) {
        x0 = _mm256_maskload_ps(x_ptr, mask6);

        x0_max = _mm256_maxwise3_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise3_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_maskstore_ps(y_ptr, mask6, z0);

        x_ptr += 6;
        y_ptr += 6;
        r -= AVX2_FLOAT_STRIDE / 4;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r * stride);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0_max = _mm256_maxwise3_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise3_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_maskstore_ps(y_ptr, mask, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride4_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0_max = _mm256_maxwise4_ps(x0);
        x1_max = _mm256_maxwise4_ps(x1);
        x2_max = _mm256_maxwise4_ps(x2);
        x3_max = _mm256_maxwise4_ps(x3);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);
        y2 = _mm256_softmaxexp_ps(x2, x2_max);
        y3 = _mm256_softmaxexp_ps(x3, x3_max);

        y0_sum = _mm256_sumwise4_ps(y0);
        y1_sum = _mm256_sumwise4_ps(y1);
        y2_sum = _mm256_sumwise4_ps(y2);
        y3_sum = _mm256_sumwise4_ps(y3);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);
        z2 = _mm256_div_ps(y2, y2_sum);
        z3 = _mm256_div_ps(y3, y3_sum);

        _mm256_stream_x4_ps(y_ptr, z0, z1, z2, z3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x0_max = _mm256_maxwise4_ps(x0);
        x1_max = _mm256_maxwise4_ps(x1);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);

        y0_sum = _mm256_sumwise4_ps(y0);
        y1_sum = _mm256_sumwise4_ps(y1);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);

        _mm256_stream_x2_ps(y_ptr, z0, z1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r >= AVX2_FLOAT_STRIDE / 4) {
        _mm256_load_x1_ps(x_ptr, x0);

        x0_max = _mm256_maxwise4_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise4_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_stream_x1_ps(y_ptr, z0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 4;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r * stride);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0_max = _mm256_maxwise4_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise4_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_maskstore_ps(y_ptr, mask, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride5to7_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride < 5 || stride > 7) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride);
    const __m256 minf = _mm256_set1_ps(-HUGE_VALF);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= 4) {
        x0 = _mm256_maskload_ps(x_ptr, mask);
        x1 = _mm256_maskload_ps(x_ptr + stride, mask);
        x2 = _mm256_maskload_ps(x_ptr + stride * 2, mask);
        x3 = _mm256_maskload_ps(x_ptr + stride * 3, mask);

        x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));
        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));
        x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));
        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

        x0_max = _mm256_maxwise8_ps(x0);
        x1_max = _mm256_maxwise8_ps(x1);
        x2_max = _mm256_maxwise8_ps(x2);
        x3_max = _mm256_maxwise8_ps(x3);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);
        y2 = _mm256_softmaxexp_ps(x2, x2_max);
        y3 = _mm256_softmaxexp_ps(x3, x3_max);

        y0_sum = _mm256_sumwise8_ps(y0);
        y1_sum = _mm256_sumwise8_ps(y1);
        y2_sum = _mm256_sumwise8_ps(y2);
        y3_sum = _mm256_sumwise8_ps(y3);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);
        z2 = _mm256_div_ps(y2, y2_sum);
        z3 = _mm256_div_ps(y3, y3_sum);

        _mm256_storeu_ps(y_ptr, z0);
        _mm256_storeu_ps(y_ptr + stride, z1);
        _mm256_storeu_ps(y_ptr + stride * 2, z2);
        _mm256_maskstore_ps(y_ptr + stride * 3, mask, z3);

        x_ptr += stride * 4;
        y_ptr += stride * 4;
        r -= 4;
    }
    if (r >= 2) {
        x0 = _mm256_maskload_ps(x_ptr, mask);
        x1 = _mm256_maskload_ps(x_ptr + stride, mask);

        x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));
        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

        x0_max = _mm256_maxwise8_ps(x0);
        x1_max = _mm256_maxwise8_ps(x1);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);

        y0_sum = _mm256_sumwise8_ps(y0);
        y1_sum = _mm256_sumwise8_ps(y1);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);

        _mm256_storeu_ps(y_ptr, z0);
        _mm256_maskstore_ps(y_ptr + stride, mask, z1);

        x_ptr += stride * 2;
        y_ptr += stride * 2;
        r -= 2;
    }
    if (r > 0) {
        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

        x0_max = _mm256_maxwise8_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise8_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_maskstore_ps(y_ptr, mask, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride8_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 8) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0_max = _mm256_maxwise8_ps(x0);
        x1_max = _mm256_maxwise8_ps(x1);
        x2_max = _mm256_maxwise8_ps(x2);
        x3_max = _mm256_maxwise8_ps(x3);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);
        y2 = _mm256_softmaxexp_ps(x2, x2_max);
        y3 = _mm256_softmaxexp_ps(x3, x3_max);

        y0_sum = _mm256_sumwise8_ps(y0);
        y1_sum = _mm256_sumwise8_ps(y1);
        y2_sum = _mm256_sumwise8_ps(y2);
        y3_sum = _mm256_sumwise8_ps(y3);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);
        z2 = _mm256_div_ps(y2, y2_sum);
        z3 = _mm256_div_ps(y3, y3_sum);

        _mm256_store_x4_ps(y_ptr, z0, z1, z2, z3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r >= AVX2_FLOAT_STRIDE / 4) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x0_max = _mm256_maxwise8_ps(x0);
        x1_max = _mm256_maxwise8_ps(x1);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);
        y1 = _mm256_softmaxexp_ps(x1, x1_max);

        y0_sum = _mm256_sumwise8_ps(y0);
        y1_sum = _mm256_sumwise8_ps(y1);

        z0 = _mm256_div_ps(y0, y0_sum);
        z1 = _mm256_div_ps(y1, y1_sum);

        _mm256_store_x2_ps(y_ptr, z0, z1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_load_x1_ps(x_ptr, x0);

        x0_max = _mm256_maxwise8_ps(x0);

        y0 = _mm256_softmaxexp_ps(x0, x0_max);

        y0_sum = _mm256_sumwise8_ps(y0);

        z0 = _mm256_div_ps(y0, y0_sum);

        _mm256_store_x1_ps(y_ptr, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride9to15_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE || stride >= AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    const __m256 minf = _mm256_set1_ps(-HUGE_VALF);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x01_max, x23_max, y01_sum, y23_sum;

    uint r = n;

    while (r >= 2) {
        _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
        _mm256_maskload_x2_ps(x_ptr + stride, x2, x3, mask);

        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));
        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

        x01_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1));
        x23_max = _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3));

        y0 = _mm256_softmaxexp_ps(x0, x01_max);
        y1 = _mm256_softmaxexp_ps(x1, x01_max);
        y2 = _mm256_softmaxexp_ps(x2, x23_max);
        y3 = _mm256_softmaxexp_ps(x3, x23_max);

        y01_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1));
        y23_sum = _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3));

        z0 = _mm256_div_ps(y0, y01_sum);
        z1 = _mm256_div_ps(y1, y01_sum);
        z2 = _mm256_div_ps(y2, y23_sum);
        z3 = _mm256_div_ps(y3, y23_sum);

        _mm256_storeu_x2_ps(y_ptr, z0, z1);
        _mm256_maskstore_x2_ps(y_ptr + stride, z2, z3, mask);

        x_ptr += stride * 2;
        y_ptr += stride * 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);

        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

        x01_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1));

        y0 = _mm256_softmaxexp_ps(x0, x01_max);
        y1 = _mm256_softmaxexp_ps(x1, x01_max);

        y01_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1));

        z0 = _mm256_div_ps(y0, y01_sum);
        z1 = _mm256_div_ps(y1, y01_sum);

        _mm256_maskstore_x2_ps(y_ptr, z0, z1, mask);
    }

    return SUCCESS;
}

int vw_softmax_stride16_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x01_max, x23_max, y01_sum, y23_sum;

    uint r = n;

    while (r >= 2) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x01_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1));
        x23_max = _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3));

        y0 = _mm256_softmaxexp_ps(x0, x01_max);
        y1 = _mm256_softmaxexp_ps(x1, x01_max);
        y2 = _mm256_softmaxexp_ps(x2, x23_max);
        y3 = _mm256_softmaxexp_ps(x3, x23_max);

        y01_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1));
        y23_sum = _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3));

        z0 = _mm256_div_ps(y0, y01_sum);
        z1 = _mm256_div_ps(y1, y01_sum);
        z2 = _mm256_div_ps(y2, y23_sum);
        z3 = _mm256_div_ps(y3, y23_sum);

        _mm256_store_x4_ps(y_ptr, z0, z1, z2, z3);

        x_ptr += stride * 2;
        y_ptr += stride * 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x01_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1));

        y0 = _mm256_softmaxexp_ps(x0, x01_max);
        y1 = _mm256_softmaxexp_ps(x1, x01_max);

        y01_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1));

        z0 = _mm256_div_ps(y0, y01_sum);
        z1 = _mm256_div_ps(y1, y01_sum);

        _mm256_store_x2_ps(y_ptr, z0, z1);
    }

    return SUCCESS;
}

int vw_softmax_stride17to23_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 2 || stride >= AVX2_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    const __m256 minf = _mm256_set1_ps(-HUGE_VALF);

    __m256 x0, x1, x2, y0, y1, y2, z0, z1, z2;
    __m256 x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);

        x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

        x_max = _mm256_max_ps(_mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)), _mm256_maxwise8_ps(x2));

        y0 = _mm256_softmaxexp_ps(x0, x_max);
        y1 = _mm256_softmaxexp_ps(x1, x_max);
        y2 = _mm256_softmaxexp_ps(x2, x_max);

        y_sum = _mm256_add_ps(_mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)), _mm256_sumwise8_ps(y2));

        y_rcp_sum = _mm256_div_ps(_mm256_set1_ps(1), y_sum);

        z0 = _mm256_mul_ps(y0, y_rcp_sum);
        z1 = _mm256_mul_ps(y1, y_rcp_sum);
        z2 = _mm256_mul_ps(y2, y_rcp_sum);

        _mm256_maskstore_x3_ps(y_ptr, z0, z1, z2, mask);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride24_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, y0, y1, y2, z0, z1, z2;
    __m256 x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);

        x_max = _mm256_max_ps(_mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)), _mm256_maxwise8_ps(x2));

        y0 = _mm256_softmaxexp_ps(x0, x_max);
        y1 = _mm256_softmaxexp_ps(x1, x_max);
        y2 = _mm256_softmaxexp_ps(x2, x_max);

        y_sum = _mm256_add_ps(_mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)), _mm256_sumwise8_ps(y2));

        y_rcp_sum = _mm256_div_ps(_mm256_set1_ps(1), y_sum);

        z0 = _mm256_mul_ps(y0, y_rcp_sum);
        z1 = _mm256_mul_ps(y1, y_rcp_sum);
        z2 = _mm256_mul_ps(y2, y_rcp_sum);

        _mm256_store_x3_ps(y_ptr, z0, z1, z2);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride25to31_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 3 || stride >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    const __m256 minf = _mm256_set1_ps(-HUGE_VALF);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);

        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

        x_max = _mm256_max_ps(
            _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
            _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3))
        );

        y0 = _mm256_softmaxexp_ps(x0, x_max);
        y1 = _mm256_softmaxexp_ps(x1, x_max);
        y2 = _mm256_softmaxexp_ps(x2, x_max);
        y3 = _mm256_softmaxexp_ps(x3, x_max);

        y_sum = _mm256_add_ps(
            _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
            _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3))
        );

        y_rcp_sum = _mm256_div_ps(_mm256_set1_ps(1), y_sum);

        z0 = _mm256_mul_ps(y0, y_rcp_sum);
        z1 = _mm256_mul_ps(y1, y_rcp_sum);
        z2 = _mm256_mul_ps(y2, y_rcp_sum);
        z3 = _mm256_mul_ps(y3, y_rcp_sum);

        _mm256_maskstore_x4_ps(y_ptr, z0, z1, z2, z3, mask);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride32_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256 x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x_max = _mm256_max_ps(
            _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
            _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3))
        );

        y0 = _mm256_softmaxexp_ps(x0, x_max);
        y1 = _mm256_softmaxexp_ps(x1, x_max);
        y2 = _mm256_softmaxexp_ps(x2, x_max);
        y3 = _mm256_softmaxexp_ps(x3, x_max);

        y_sum = _mm256_add_ps(
            _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
            _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3))
        );

        y_rcp_sum = _mm256_div_ps(_mm256_set1_ps(1), y_sum);

        z0 = _mm256_mul_ps(y0, y_rcp_sum);
        z1 = _mm256_mul_ps(y1, y_rcp_sum);
        z2 = _mm256_mul_ps(y2, y_rcp_sum);
        z3 = _mm256_mul_ps(y3, y_rcp_sum);

        _mm256_store_x4_ps(y_ptr, z0, z1, z2, z3);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride32x_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride % (AVX2_FLOAT_STRIDE * 4) != 0) || (stride <= AVX2_FLOAT_STRIDE * 4)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint r;
    infloats xc_ptr;
    outfloats yc_ptr;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    for (uint i = 0; i < n; i++) {
        __m256 x_max = _mm256_undefined_ps(), y_sum = _mm256_undefined_ps();

        r = stride;
        xc_ptr = x_ptr;

        {
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_ps(
                _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
                _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3))
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_ps(
                _mm256_max_ps(
                    _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
                    _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3))
                ),
                x_max
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }

        r = stride;
        xc_ptr = x_ptr;
        yc_ptr = y_ptr;

        {
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_ps(x0, x_max);
            y1 = _mm256_softmaxexp_ps(x1, x_max);
            y2 = _mm256_softmaxexp_ps(x2, x_max);
            y3 = _mm256_softmaxexp_ps(x3, x_max);

            _mm256_store_x4_ps(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_ps(
                _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
                _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3))
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            yc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_ps(x0, x_max);
            y1 = _mm256_softmaxexp_ps(x1, x_max);
            y2 = _mm256_softmaxexp_ps(x2, x_max);
            y3 = _mm256_softmaxexp_ps(x3, x_max);

            _mm256_store_x4_ps(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
                    _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3))
                ),
                y_sum
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            yc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }

        r = stride;
        yc_ptr = y_ptr;

        __m256 y_rcp_sum = _mm256_div_ps(_mm256_set1_ps(1), y_sum);

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);
            z1 = _mm256_mul_ps(y1, y_rcp_sum);
            z2 = _mm256_mul_ps(y2, y_rcp_sum);
            z3 = _mm256_mul_ps(y3, y_rcp_sum);

            _mm256_store_x4_ps(yc_ptr, z0, z1, z2, z3);

            yc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }

        x_ptr += stride;
        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_softmax_aligned_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride % AVX2_FLOAT_STRIDE != 0) || (stride <= AVX2_FLOAT_STRIDE * 4)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 minf = _mm256_set1_ps(-HUGE_VALF);

    uint r;
    infloats xc_ptr;
    outfloats yc_ptr;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    for (uint i = 0; i < n; i++) {
        __m256 x_max = minf, y_sum = _mm256_setzero_ps();

        r = stride;
        xc_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_ps(
                _mm256_max_ps(
                    _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
                    _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3))
                ),
                x_max
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(xc_ptr, x0, x1);

            x_max = _mm256_max_ps(
                _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
                x_max
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(xc_ptr, x0);

            x_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), x_max);
        }

        r = stride;
        xc_ptr = x_ptr;
        yc_ptr = y_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_ps(x0, x_max);
            y1 = _mm256_softmaxexp_ps(x1, x_max);
            y2 = _mm256_softmaxexp_ps(x2, x_max);
            y3 = _mm256_softmaxexp_ps(x3, x_max);

            _mm256_store_x4_ps(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
                    _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3))
                ),
                y_sum
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            yc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(xc_ptr, x0, x1);

            y0 = _mm256_softmaxexp_ps(x0, x_max);
            y1 = _mm256_softmaxexp_ps(x1, x_max);

            _mm256_store_x2_ps(yc_ptr, y0, y1);

            y_sum = _mm256_add_ps(
                _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
                y_sum
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 2;
            yc_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(xc_ptr, x0);

            y0 = _mm256_softmaxexp_ps(x0, x_max);

            _mm256_store_x1_ps(yc_ptr, y0);

            y_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), y_sum);
        }

        r = stride;
        yc_ptr = y_ptr;

        __m256 y_rcp_sum = _mm256_div_ps(_mm256_set1_ps(1), y_sum);

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);
            z1 = _mm256_mul_ps(y1, y_rcp_sum);
            z2 = _mm256_mul_ps(y2, y_rcp_sum);
            z3 = _mm256_mul_ps(y3, y_rcp_sum);

            _mm256_store_x4_ps(yc_ptr, z0, z1, z2, z3);

            yc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(yc_ptr, y0, y1);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);
            z1 = _mm256_mul_ps(y1, y_rcp_sum);

            _mm256_store_x2_ps(yc_ptr, z0, z1);

            yc_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(yc_ptr, y0);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);

            _mm256_store_x1_ps(yc_ptr, z0);
        }

        x_ptr += stride;
        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_softmax_unaligned_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride % AVX2_FLOAT_STRIDE == 0) || (stride <= AVX2_FLOAT_STRIDE * 4)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    const __m256 minf = _mm256_set1_ps(-HUGE_VALF);

    uint r;
    infloats xc_ptr;
    outfloats yc_ptr;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    for (uint i = 0; i < n; i++) {
        __m256 x_max = minf, y_sum = _mm256_setzero_ps();

        r = stride;
        xc_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_ps(
                _mm256_max_ps(
                    _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
                    _mm256_max_ps(_mm256_maxwise8_ps(x2), _mm256_maxwise8_ps(x3))
                ),
                x_max
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(xc_ptr, x0, x1);

            x_max = _mm256_max_ps(
                _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1)),
                x_max
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(xc_ptr, x0);

            x_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), x_max);

            xc_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_ps(xc_ptr, x0, mask);

            x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

            x_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), x_max);
        }

        r = stride;
        xc_ptr = x_ptr;
        yc_ptr = y_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_ps(x0, x_max);
            y1 = _mm256_softmaxexp_ps(x1, x_max);
            y2 = _mm256_softmaxexp_ps(x2, x_max);
            y3 = _mm256_softmaxexp_ps(x3, x_max);

            _mm256_storeu_x4_ps(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
                    _mm256_add_ps(_mm256_sumwise8_ps(y2), _mm256_sumwise8_ps(y3))
                ),
                y_sum
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            yc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(xc_ptr, x0, x1);

            y0 = _mm256_softmaxexp_ps(x0, x_max);
            y1 = _mm256_softmaxexp_ps(x1, x_max);

            _mm256_storeu_x2_ps(yc_ptr, y0, y1);

            y_sum = _mm256_add_ps(
                _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1)),
                y_sum
            );

            xc_ptr += AVX2_FLOAT_STRIDE * 2;
            yc_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(xc_ptr, x0);

            y0 = _mm256_softmaxexp_ps(x0, x_max);

            _mm256_storeu_x1_ps(yc_ptr, y0);

            y_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), y_sum);

            xc_ptr += AVX2_FLOAT_STRIDE;
            yc_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_ps(xc_ptr, x0, mask);

            x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

            y0 = _mm256_softmaxexp_ps(x0, x_max);

            _mm256_maskstore_x1_ps(yc_ptr, y0, mask);

            y_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), y_sum);
        }

        r = stride;
        yc_ptr = y_ptr;

        __m256 y_rcp_sum = _mm256_div_ps(_mm256_set1_ps(1), y_sum);

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(yc_ptr, y0, y1, y2, y3);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);
            z1 = _mm256_mul_ps(y1, y_rcp_sum);
            z2 = _mm256_mul_ps(y2, y_rcp_sum);
            z3 = _mm256_mul_ps(y3, y_rcp_sum);

            _mm256_storeu_x4_ps(yc_ptr, z0, z1, z2, z3);

            yc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(yc_ptr, y0, y1);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);
            z1 = _mm256_mul_ps(y1, y_rcp_sum);

            _mm256_storeu_x2_ps(yc_ptr, z0, z1);

            yc_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(yc_ptr, y0);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);

            _mm256_storeu_x1_ps(yc_ptr, z0);

            yc_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_ps(yc_ptr, y0, mask);

            z0 = _mm256_mul_ps(y0, y_rcp_sum);

            _mm256_maskstore_x1_ps(yc_ptr, z0, mask);
        }

        x_ptr += stride;
        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_softmax_strideleq8_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride == 1) {
        return vw_softmax_stride1_s(n, stride, x_ptr, y_ptr);
    }
    else if (stride == 2) {
        return vw_softmax_stride2_s(n, stride, x_ptr, y_ptr);
    }
    else if (stride == 3) {
        return vw_softmax_stride3_s(n, stride, x_ptr, y_ptr);
    }
    else if (stride == 4) {
        return vw_softmax_stride4_s(n, stride, x_ptr, y_ptr);
    }
    else if (stride <= 7) {
        return vw_softmax_stride5to7_s(n, stride, x_ptr, y_ptr);
    }
    else if (stride == AVX2_FLOAT_STRIDE) {
        return vw_softmax_stride8_s(n, stride, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int vw_softmax_stride_aligned_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride == AVX2_FLOAT_STRIDE) {
        return vw_softmax_stride8_s(n, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 2) {
        return vw_softmax_stride16_s(n, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 3) {
        return vw_softmax_stride24_s(n, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 4) {
        return vw_softmax_stride32_s(n, stride, x_ptr, y_ptr);
    }
    if (stride % (AVX2_FLOAT_STRIDE * 4) == 0) {
        return vw_softmax_stride32x_s(n, stride, x_ptr, y_ptr);
    }

    return vw_softmax_aligned_s(n, stride, x_ptr, y_ptr);
}

int vw_softmax_stride_unaligned_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride <= AVX2_FLOAT_STRIDE) {
        return vw_softmax_strideleq8_s(n, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 2) {
        return vw_softmax_stride9to15_s(n, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 3) {
        return vw_softmax_stride17to23_s(n, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 4) {
        return vw_softmax_stride25to31_s(n, stride, x_ptr, y_ptr);
    }

    return vw_softmax_unaligned_s(n, stride, x_ptr, y_ptr);
}

#pragma managed

void AvxBlas::Vectorwise::Softmax(UInt32 n, UInt32 stride, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, stride);

    Util::CheckLength(n * stride, x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = vw_softmax_stride_aligned_s(n, stride, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = vw_softmax_stride_unaligned_s(n, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}