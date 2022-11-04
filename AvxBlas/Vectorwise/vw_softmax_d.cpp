#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_d.hpp"

#include <math.h>

using namespace System;

#pragma unmanaged

__forceinline __m256d _mm256_softmaxexp_pd(__m256d x, __m256d x_max) {
    __m256d y = _mm256_sub_pd(x, x_max);
    __m256d z = _mm256_exp_pd(y);

    return z;
}

__forceinline __m256d _mm256_maxwise2_pd(__m256d x) {
    __m256d y = _mm256_max_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_CDAB));

    return y;
}

__forceinline __m256d _mm256_sumwise2_pd(__m256d x) {
    __m256d y = _mm256_add_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_CDAB));

    return y;
}

__forceinline __m256d _mm256_maxwise3_pd(__m256d x) {
    __m256d y = _mm256_max_pd(_mm256_max_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_DBAC)), _mm256_permute4x64_pd(x, _MM_PERM_DACB));

    return y;
}

__forceinline __m256d _mm256_sumwise3_pd(__m256d x) {
    __m256d y = _mm256_add_pd(_mm256_add_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_DBAC)), _mm256_permute4x64_pd(x, _MM_PERM_DACB));

    return y;
}

__forceinline __m256d _mm256_maxwise4_pd(__m256d x) {
    __m256d y = _mm256_max_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_CDAB));
    __m256d z = _mm256_max_pd(y, _mm256_permute4x64_pd(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256d _mm256_sumwise4_pd(__m256d x) {
    __m256d y = _mm256_add_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_CDAB));
    __m256d z = _mm256_add_pd(y, _mm256_permute4x64_pd(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256d _mm256_normal_asone_pd(__m256d x) {
    __m256d y = _mm256_and_pd(_mm256_set1_pd(NAN), _mm256_cmp_pd(x, x, _CMP_NEQ_UQ));
    __m256d z = _mm256_add_pd(_mm256_set1_pd(1), y);

    return z;
}

int vw_softmax_stride1_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride != 1 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d fills = _mm256_set1_pd(1);

    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        y0 = _mm256_normal_asone_pd(x0);
        y1 = _mm256_normal_asone_pd(x1);
        y2 = _mm256_normal_asone_pd(x2);
        y3 = _mm256_normal_asone_pd(x3);

        _mm256_stream_x4_pd(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        y0 = _mm256_normal_asone_pd(x0);
        y1 = _mm256_normal_asone_pd(x1);

        _mm256_stream_x2_pd(y_ptr, y0, y1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x1_pd(x_ptr, x0);

        y0 = _mm256_normal_asone_pd(x0);

        _mm256_stream_x1_pd(y_ptr, y0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        x0 = _mm256_maskload_pd(x_ptr, mask);

        y0 = _mm256_normal_asone_pd(x0);

        _mm256_maskstore_pd(y_ptr, mask, y0);
    }

    return SUCCESS;
}

int vw_softmax_stride2_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride != 2 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256d x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x0_max = _mm256_maxwise2_pd(x0);
        x1_max = _mm256_maxwise2_pd(x1);
        x2_max = _mm256_maxwise2_pd(x2);
        x3_max = _mm256_maxwise2_pd(x3);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);
        y1 = _mm256_softmaxexp_pd(x1, x1_max);
        y2 = _mm256_softmaxexp_pd(x2, x2_max);
        y3 = _mm256_softmaxexp_pd(x3, x3_max);

        y0_sum = _mm256_sumwise2_pd(y0);
        y1_sum = _mm256_sumwise2_pd(y1);
        y2_sum = _mm256_sumwise2_pd(y2);
        y3_sum = _mm256_sumwise2_pd(y3);

        z0 = _mm256_div_pd(y0, y0_sum);
        z1 = _mm256_div_pd(y1, y1_sum);
        z2 = _mm256_div_pd(y2, y2_sum);
        z3 = _mm256_div_pd(y3, y3_sum);

        _mm256_stream_x4_pd(y_ptr, z0, z1, z2, z3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        x0_max = _mm256_maxwise2_pd(x0);
        x1_max = _mm256_maxwise2_pd(x1);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);
        y1 = _mm256_softmaxexp_pd(x1, x1_max);

        y0_sum = _mm256_sumwise2_pd(y0);
        y1_sum = _mm256_sumwise2_pd(y1);

        z0 = _mm256_div_pd(y0, y0_sum);
        z1 = _mm256_div_pd(y1, y1_sum);

        _mm256_stream_x2_pd(y_ptr, z0, z1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x1_pd(x_ptr, x0);

        x0_max = _mm256_maxwise2_pd(x0);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);

        y0_sum = _mm256_sumwise2_pd(y0);

        z0 = _mm256_div_pd(y0, y0_sum);

        _mm256_stream_x1_pd(y_ptr, z0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r * stride);

        x0 = _mm256_maskload_pd(x_ptr, mask);

        x0_max = _mm256_maxwise2_pd(x0);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);

        y0_sum = _mm256_sumwise2_pd(y0);

        z0 = _mm256_div_pd(y0, y0_sum);

        _mm256_maskstore_pd(y_ptr, mask, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride3_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride != 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask3 = _mm256_setmask_pd(3);

    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256d x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE) {
        x0 = _mm256_maskload_pd(x_ptr, mask3);
        x1 = _mm256_maskload_pd(x_ptr + 3, mask3);
        x2 = _mm256_maskload_pd(x_ptr + 6, mask3);
        x3 = _mm256_maskload_pd(x_ptr + 9, mask3);

        x0_max = _mm256_maxwise3_pd(x0);
        x1_max = _mm256_maxwise3_pd(x1);
        x2_max = _mm256_maxwise3_pd(x2);
        x3_max = _mm256_maxwise3_pd(x3);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);
        y1 = _mm256_softmaxexp_pd(x1, x1_max);
        y2 = _mm256_softmaxexp_pd(x2, x2_max);
        y3 = _mm256_softmaxexp_pd(x3, x3_max);

        y0_sum = _mm256_sumwise3_pd(y0);
        y1_sum = _mm256_sumwise3_pd(y1);
        y2_sum = _mm256_sumwise3_pd(y2);
        y3_sum = _mm256_sumwise3_pd(y3);

        z0 = _mm256_div_pd(y0, y0_sum);
        z1 = _mm256_div_pd(y1, y1_sum);
        z2 = _mm256_div_pd(y2, y2_sum);
        z3 = _mm256_div_pd(y3, y3_sum);

        _mm256_storeu_pd(y_ptr, z0);
        _mm256_storeu_pd(y_ptr + 3, z1);
        _mm256_storeu_pd(y_ptr + 6, z2);
        _mm256_maskstore_pd(y_ptr + 9, mask3, z3);

        x_ptr += 12;
        y_ptr += 12;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        x0 = _mm256_maskload_pd(x_ptr, mask3);
        x1 = _mm256_maskload_pd(x_ptr + 3, mask3);

        x0_max = _mm256_maxwise3_pd(x0);
        x1_max = _mm256_maxwise3_pd(x1);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);
        y1 = _mm256_softmaxexp_pd(x1, x1_max);

        y0_sum = _mm256_sumwise3_pd(y0);
        y1_sum = _mm256_sumwise3_pd(y1);

        z0 = _mm256_div_pd(y0, y0_sum);
        z1 = _mm256_div_pd(y1, y1_sum);

        _mm256_storeu_pd(y_ptr, z0);
        _mm256_maskstore_pd(y_ptr + 3, mask3, z1);

        x_ptr += 6;
        y_ptr += 6;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        x0 = _mm256_maskload_pd(x_ptr, mask3);

        x0_max = _mm256_maxwise3_pd(x0);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);

        y0_sum = _mm256_sumwise3_pd(y0);

        z0 = _mm256_div_pd(y0, y0_sum);

        _mm256_maskstore_pd(y_ptr, mask3, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride4_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256d x0_max, x1_max, x2_max, x3_max, y0_sum, y1_sum, y2_sum, y3_sum;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x0_max = _mm256_maxwise4_pd(x0);
        x1_max = _mm256_maxwise4_pd(x1);
        x2_max = _mm256_maxwise4_pd(x2);
        x3_max = _mm256_maxwise4_pd(x3);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);
        y1 = _mm256_softmaxexp_pd(x1, x1_max);
        y2 = _mm256_softmaxexp_pd(x2, x2_max);
        y3 = _mm256_softmaxexp_pd(x3, x3_max);

        y0_sum = _mm256_sumwise4_pd(y0);
        y1_sum = _mm256_sumwise4_pd(y1);
        y2_sum = _mm256_sumwise4_pd(y2);
        y3_sum = _mm256_sumwise4_pd(y3);

        z0 = _mm256_div_pd(y0, y0_sum);
        z1 = _mm256_div_pd(y1, y1_sum);
        z2 = _mm256_div_pd(y2, y2_sum);
        z3 = _mm256_div_pd(y3, y3_sum);

        _mm256_stream_x4_pd(y_ptr, z0, z1, z2, z3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        x0_max = _mm256_maxwise4_pd(x0);
        x1_max = _mm256_maxwise4_pd(x1);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);
        y1 = _mm256_softmaxexp_pd(x1, x1_max);

        y0_sum = _mm256_sumwise4_pd(y0);
        y1_sum = _mm256_sumwise4_pd(y1);

        z0 = _mm256_div_pd(y0, y0_sum);
        z1 = _mm256_div_pd(y1, y1_sum);

        _mm256_stream_x2_pd(y_ptr, z0, z1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        _mm256_load_x1_pd(x_ptr, x0);

        x0_max = _mm256_maxwise4_pd(x0);

        y0 = _mm256_softmaxexp_pd(x0, x0_max);

        y0_sum = _mm256_sumwise4_pd(y0);

        z0 = _mm256_div_pd(y0, y0_sum);

        _mm256_stream_x1_pd(y_ptr, z0);
    }

    return SUCCESS;
}

int vw_softmax_stride5to7_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE || stride >= AVX2_DOUBLE_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256d x01_max, x23_max, y01_sum, y23_sum;

    uint r = n;

    while (r >= 2) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);
        _mm256_maskload_x2_pd(x_ptr + stride, x2, x3, mask);

        x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));
        x3 = _mm256_or_pd(x3, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

        x01_max = _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1));
        x23_max = _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3));

        y0 = _mm256_softmaxexp_pd(x0, x01_max);
        y1 = _mm256_softmaxexp_pd(x1, x01_max);
        y2 = _mm256_softmaxexp_pd(x2, x23_max);
        y3 = _mm256_softmaxexp_pd(x3, x23_max);

        y01_sum = _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1));
        y23_sum = _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3));

        z0 = _mm256_div_pd(y0, y01_sum);
        z1 = _mm256_div_pd(y1, y01_sum);
        z2 = _mm256_div_pd(y2, y23_sum);
        z3 = _mm256_div_pd(y3, y23_sum);

        _mm256_storeu_x2_pd(y_ptr, z0, z1);
        _mm256_maskstore_x2_pd(y_ptr + stride, z2, z3, mask);

        x_ptr += stride * 2;
        y_ptr += stride * 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

        x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

        x01_max = _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1));

        y0 = _mm256_softmaxexp_pd(x0, x01_max);
        y1 = _mm256_softmaxexp_pd(x1, x01_max);

        y01_sum = _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1));

        z0 = _mm256_div_pd(y0, y01_sum);
        z1 = _mm256_div_pd(y1, y01_sum);

        _mm256_maskstore_x2_pd(y_ptr, z0, z1, mask);
    }

    return SUCCESS;
}

int vw_softmax_stride8_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256d x01_max, x23_max, y01_sum, y23_sum;

    uint r = n;

    while (r >= 2) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x01_max = _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1));
        x23_max = _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3));

        y0 = _mm256_softmaxexp_pd(x0, x01_max);
        y1 = _mm256_softmaxexp_pd(x1, x01_max);
        y2 = _mm256_softmaxexp_pd(x2, x23_max);
        y3 = _mm256_softmaxexp_pd(x3, x23_max);

        y01_sum = _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1));
        y23_sum = _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3));

        z0 = _mm256_div_pd(y0, y01_sum);
        z1 = _mm256_div_pd(y1, y01_sum);
        z2 = _mm256_div_pd(y2, y23_sum);
        z3 = _mm256_div_pd(y3, y23_sum);

        _mm256_store_x4_pd(y_ptr, z0, z1, z2, z3);

        x_ptr += stride * 2;
        y_ptr += stride * 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        x01_max = _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1));

        y0 = _mm256_softmaxexp_pd(x0, x01_max);
        y1 = _mm256_softmaxexp_pd(x1, x01_max);

        y01_sum = _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1));

        z0 = _mm256_div_pd(y0, y01_sum);
        z1 = _mm256_div_pd(y1, y01_sum);

        _mm256_store_x2_pd(y_ptr, z0, z1);
    }

    return SUCCESS;
}

int vw_softmax_stride9to11_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 2 || stride >= AVX2_DOUBLE_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    __m256d x0, x1, x2, y0, y1, y2, z0, z1, z2;
    __m256d x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

        x2 = _mm256_or_pd(x2, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

        x_max = _mm256_max_pd(_mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)), _mm256_maxwise4_pd(x2));

        y0 = _mm256_softmaxexp_pd(x0, x_max);
        y1 = _mm256_softmaxexp_pd(x1, x_max);
        y2 = _mm256_softmaxexp_pd(x2, x_max);

        y_sum = _mm256_add_pd(_mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)), _mm256_sumwise4_pd(y2));

        y_rcp_sum = _mm256_div_pd(_mm256_set1_pd(1), y_sum);

        z0 = _mm256_mul_pd(y0, y_rcp_sum);
        z1 = _mm256_mul_pd(y1, y_rcp_sum);
        z2 = _mm256_mul_pd(y2, y_rcp_sum);

        _mm256_maskstore_x3_pd(y_ptr, z0, z1, z2, mask);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride12_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, y0, y1, y2, z0, z1, z2;
    __m256d x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_load_x3_pd(x_ptr, x0, x1, x2);

        x_max = _mm256_max_pd(_mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)), _mm256_maxwise4_pd(x2));

        y0 = _mm256_softmaxexp_pd(x0, x_max);
        y1 = _mm256_softmaxexp_pd(x1, x_max);
        y2 = _mm256_softmaxexp_pd(x2, x_max);

        y_sum = _mm256_add_pd(_mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)), _mm256_sumwise4_pd(y2));

        y_rcp_sum = _mm256_div_pd(_mm256_set1_pd(1), y_sum);

        z0 = _mm256_mul_pd(y0, y_rcp_sum);
        z1 = _mm256_mul_pd(y1, y_rcp_sum);
        z2 = _mm256_mul_pd(y2, y_rcp_sum);

        _mm256_store_x3_pd(y_ptr, z0, z1, z2);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride13to15_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 3 || stride >= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256d x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

        x3 = _mm256_or_pd(x3, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

        x_max = _mm256_max_pd(
            _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
            _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3))
        );

        y0 = _mm256_softmaxexp_pd(x0, x_max);
        y1 = _mm256_softmaxexp_pd(x1, x_max);
        y2 = _mm256_softmaxexp_pd(x2, x_max);
        y3 = _mm256_softmaxexp_pd(x3, x_max);

        y_sum = _mm256_add_pd(
            _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
            _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3))
        );

        y_rcp_sum = _mm256_div_pd(_mm256_set1_pd(1), y_sum);

        z0 = _mm256_mul_pd(y0, y_rcp_sum);
        z1 = _mm256_mul_pd(y1, y_rcp_sum);
        z2 = _mm256_mul_pd(y2, y_rcp_sum);
        z3 = _mm256_mul_pd(y3, y_rcp_sum);

        _mm256_maskstore_x4_pd(y_ptr, z0, z1, z2, z3, mask);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride16_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
    __m256d x_max, y_sum, y_rcp_sum;

    uint r = n;

    while (r > 0) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x_max = _mm256_max_pd(
            _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
            _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3))
        );

        y0 = _mm256_softmaxexp_pd(x0, x_max);
        y1 = _mm256_softmaxexp_pd(x1, x_max);
        y2 = _mm256_softmaxexp_pd(x2, x_max);
        y3 = _mm256_softmaxexp_pd(x3, x_max);

        y_sum = _mm256_add_pd(
            _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
            _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3))
        );

        y_rcp_sum = _mm256_div_pd(_mm256_set1_pd(1), y_sum);

        z0 = _mm256_mul_pd(y0, y_rcp_sum);
        z1 = _mm256_mul_pd(y1, y_rcp_sum);
        z2 = _mm256_mul_pd(y2, y_rcp_sum);
        z3 = _mm256_mul_pd(y3, y_rcp_sum);

        _mm256_store_x4_pd(y_ptr, z0, z1, z2, z3);

        x_ptr += stride;
        y_ptr += stride;
        r--;
    }

    return SUCCESS;
}

int vw_softmax_stride16x_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride % (AVX2_DOUBLE_STRIDE * 4) != 0) || (stride <= AVX2_DOUBLE_STRIDE * 4)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint r;
    indoubles xc_ptr;
    outdoubles yc_ptr;
    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    for (uint i = 0; i < n; i++) {
        __m256d x_max = _mm256_undefined_pd(), y_sum = _mm256_undefined_pd();

        r = stride;
        xc_ptr = x_ptr;

        {
            _mm256_load_x4_pd(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_pd(
                _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
                _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3))
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_pd(
                _mm256_max_pd(
                    _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
                    _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3))
                ),
                x_max
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }

        r = stride;
        xc_ptr = x_ptr;
        yc_ptr = y_ptr;

        {
            _mm256_load_x4_pd(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_pd(x0, x_max);
            y1 = _mm256_softmaxexp_pd(x1, x_max);
            y2 = _mm256_softmaxexp_pd(x2, x_max);
            y3 = _mm256_softmaxexp_pd(x3, x_max);

            _mm256_store_x4_pd(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_pd(
                _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
                _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3))
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            yc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_pd(x0, x_max);
            y1 = _mm256_softmaxexp_pd(x1, x_max);
            y2 = _mm256_softmaxexp_pd(x2, x_max);
            y3 = _mm256_softmaxexp_pd(x3, x_max);

            _mm256_store_x4_pd(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
                    _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3))
                ),
                y_sum
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            yc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }

        r = stride;
        yc_ptr = y_ptr;

        __m256d y_rcp_sum = _mm256_div_pd(_mm256_set1_pd(1), y_sum);

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(yc_ptr, y0, y1, y2, y3);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);
            z1 = _mm256_mul_pd(y1, y_rcp_sum);
            z2 = _mm256_mul_pd(y2, y_rcp_sum);
            z3 = _mm256_mul_pd(y3, y_rcp_sum);

            _mm256_store_x4_pd(yc_ptr, z0, z1, z2, z3);

            yc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }

        x_ptr += stride;
        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_softmax_aligned_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride % AVX2_DOUBLE_STRIDE != 0) || (stride <= AVX2_DOUBLE_STRIDE * 4)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    uint r;
    indoubles xc_ptr;
    outdoubles yc_ptr;
    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    for (uint i = 0; i < n; i++) {
        __m256d x_max = minf, y_sum = _mm256_setzero_pd();

        r = stride;
        xc_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_pd(
                _mm256_max_pd(
                    _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
                    _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3))
                ),
                x_max
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(xc_ptr, x0, x1);

            x_max = _mm256_max_pd(
                _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
                x_max
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(xc_ptr, x0);

            x_max = _mm256_max_pd(_mm256_maxwise4_pd(x0), x_max);
        }

        r = stride;
        xc_ptr = x_ptr;
        yc_ptr = y_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_pd(x0, x_max);
            y1 = _mm256_softmaxexp_pd(x1, x_max);
            y2 = _mm256_softmaxexp_pd(x2, x_max);
            y3 = _mm256_softmaxexp_pd(x3, x_max);

            _mm256_store_x4_pd(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
                    _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3))
                ),
                y_sum
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            yc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(xc_ptr, x0, x1);

            y0 = _mm256_softmaxexp_pd(x0, x_max);
            y1 = _mm256_softmaxexp_pd(x1, x_max);

            _mm256_store_x2_pd(yc_ptr, y0, y1);

            y_sum = _mm256_add_pd(
                _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
                y_sum
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 2;
            yc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(xc_ptr, x0);

            y0 = _mm256_softmaxexp_pd(x0, x_max);

            _mm256_store_x1_pd(yc_ptr, y0);

            y_sum = _mm256_add_pd(_mm256_sumwise4_pd(y0), y_sum);
        }

        r = stride;
        yc_ptr = y_ptr;

        __m256d y_rcp_sum = _mm256_div_pd(_mm256_set1_pd(1), y_sum);

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(yc_ptr, y0, y1, y2, y3);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);
            z1 = _mm256_mul_pd(y1, y_rcp_sum);
            z2 = _mm256_mul_pd(y2, y_rcp_sum);
            z3 = _mm256_mul_pd(y3, y_rcp_sum);

            _mm256_store_x4_pd(yc_ptr, z0, z1, z2, z3);

            yc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(yc_ptr, y0, y1);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);
            z1 = _mm256_mul_pd(y1, y_rcp_sum);

            _mm256_store_x2_pd(yc_ptr, z0, z1);

            yc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(yc_ptr, y0);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);

            _mm256_store_x1_pd(yc_ptr, z0);
        }

        x_ptr += stride;
        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_softmax_unaligned_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride % AVX2_DOUBLE_STRIDE == 0) || (stride <= AVX2_DOUBLE_STRIDE * 4)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    uint r;
    indoubles xc_ptr;
    outdoubles yc_ptr;
    __m256d x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;

    for (uint i = 0; i < n; i++) {
        __m256d x_max = minf, y_sum = _mm256_setzero_pd();

        r = stride;
        xc_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(xc_ptr, x0, x1, x2, x3);

            x_max = _mm256_max_pd(
                _mm256_max_pd(
                    _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
                    _mm256_max_pd(_mm256_maxwise4_pd(x2), _mm256_maxwise4_pd(x3))
                ),
                x_max
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(xc_ptr, x0, x1);

            x_max = _mm256_max_pd(
                _mm256_max_pd(_mm256_maxwise4_pd(x0), _mm256_maxwise4_pd(x1)),
                x_max
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(xc_ptr, x0);

            x_max = _mm256_max_pd(_mm256_maxwise4_pd(x0), x_max);

            xc_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_pd(xc_ptr, x0, mask);

            x0 = _mm256_or_pd(x0, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            x_max = _mm256_max_pd(_mm256_maxwise4_pd(x0), x_max);
        }

        r = stride;
        xc_ptr = x_ptr;
        yc_ptr = y_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(xc_ptr, x0, x1, x2, x3);

            y0 = _mm256_softmaxexp_pd(x0, x_max);
            y1 = _mm256_softmaxexp_pd(x1, x_max);
            y2 = _mm256_softmaxexp_pd(x2, x_max);
            y3 = _mm256_softmaxexp_pd(x3, x_max);

            _mm256_storeu_x4_pd(yc_ptr, y0, y1, y2, y3);

            y_sum = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
                    _mm256_add_pd(_mm256_sumwise4_pd(y2), _mm256_sumwise4_pd(y3))
                ),
                y_sum
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 4;
            yc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(xc_ptr, x0, x1);

            y0 = _mm256_softmaxexp_pd(x0, x_max);
            y1 = _mm256_softmaxexp_pd(x1, x_max);

            _mm256_storeu_x2_pd(yc_ptr, y0, y1);

            y_sum = _mm256_add_pd(
                _mm256_add_pd(_mm256_sumwise4_pd(y0), _mm256_sumwise4_pd(y1)),
                y_sum
            );

            xc_ptr += AVX2_DOUBLE_STRIDE * 2;
            yc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(xc_ptr, x0);

            y0 = _mm256_softmaxexp_pd(x0, x_max);

            _mm256_storeu_x1_pd(yc_ptr, y0);

            y_sum = _mm256_add_pd(_mm256_sumwise4_pd(y0), y_sum);

            xc_ptr += AVX2_DOUBLE_STRIDE;
            yc_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_pd(xc_ptr, x0, mask);

            x0 = _mm256_or_pd(x0, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            y0 = _mm256_softmaxexp_pd(x0, x_max);

            _mm256_maskstore_x1_pd(yc_ptr, y0, mask);

            y_sum = _mm256_add_pd(_mm256_sumwise4_pd(y0), y_sum);
        }

        r = stride;
        yc_ptr = y_ptr;

        __m256d y_rcp_sum = _mm256_div_pd(_mm256_set1_pd(1), y_sum);

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(yc_ptr, y0, y1, y2, y3);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);
            z1 = _mm256_mul_pd(y1, y_rcp_sum);
            z2 = _mm256_mul_pd(y2, y_rcp_sum);
            z3 = _mm256_mul_pd(y3, y_rcp_sum);

            _mm256_storeu_x4_pd(yc_ptr, z0, z1, z2, z3);

            yc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(yc_ptr, y0, y1);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);
            z1 = _mm256_mul_pd(y1, y_rcp_sum);

            _mm256_storeu_x2_pd(yc_ptr, z0, z1);

            yc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(yc_ptr, y0);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);

            _mm256_storeu_x1_pd(yc_ptr, z0);

            yc_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_pd(yc_ptr, y0, mask);

            z0 = _mm256_mul_pd(y0, y_rcp_sum);

            _mm256_maskstore_x1_pd(yc_ptr, z0, mask);
        }

        x_ptr += stride;
        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_softmax_strideleq8_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride == 1) {
        return vw_softmax_stride1_d(n, stride, x_ptr, y_ptr);
    }
    else if (stride == 2) {
        return vw_softmax_stride2_d(n, stride, x_ptr, y_ptr);
    }
    else if (stride == 3) {
        return vw_softmax_stride3_d(n, stride, x_ptr, y_ptr);
    }
    else if (stride == 4) {
        return vw_softmax_stride4_d(n, stride, x_ptr, y_ptr);
    }
    else if (stride <= 7) {
        return vw_softmax_stride5to7_d(n, stride, x_ptr, y_ptr);
    }
    else if (stride == AVX2_DOUBLE_STRIDE) {
        return vw_softmax_stride8_d(n, stride, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int vw_softmax_stride_aligned_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride == AVX2_DOUBLE_STRIDE) {
        return vw_softmax_stride4_d(n, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 2) {
        return vw_softmax_stride8_d(n, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 3) {
        return vw_softmax_stride12_d(n, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 4) {
        return vw_softmax_stride16_d(n, stride, x_ptr, y_ptr);
    }
    if (stride % (AVX2_DOUBLE_STRIDE * 4) == 0) {
        return vw_softmax_stride16x_d(n, stride, x_ptr, y_ptr);
    }

    return vw_softmax_aligned_d(n, stride, x_ptr, y_ptr);
}

int vw_softmax_stride_unaligned_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride <= AVX2_DOUBLE_STRIDE * 2) {
        return vw_softmax_strideleq8_d(n, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 3) {
        return vw_softmax_stride9to11_d(n, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 4) {
        return vw_softmax_stride13to15_d(n, stride, x_ptr, y_ptr);
    }

    return vw_softmax_unaligned_d(n, stride, x_ptr, y_ptr);
}

#pragma managed

void AvxBlas::Vectorwise::Softmax(UInt32 n, UInt32 stride, Array<double>^ x, Array<double>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, stride);

    Util::CheckLength(n * stride, x, y);

    const double* x_ptr = (const double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = vw_softmax_stride_aligned_d(n, stride, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = vw_softmax_stride_unaligned_d(n, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}