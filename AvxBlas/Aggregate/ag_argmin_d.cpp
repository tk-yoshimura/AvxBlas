#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_min_d.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_cmp_d.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"
#include "../Inline/inline_loadstore_xn_epi32.hpp"
#include <memory.h>
#include <math.h>

using namespace System;

#pragma unmanaged

int ag_argmin_samples2_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, s0, s1, s2, s3;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        s0 = _mm256_minwise2_pd(x0);
        s1 = _mm256_minwise2_pd(x1);
        s2 = _mm256_minwise2_pd(x2);
        s3 = _mm256_minwise2_pd(x3);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s1));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s2));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s3));

        __m256i y = _mm256_setr_epi32(
            bsf(i0), bsf(i0 >> 2), bsf(i1), bsf(i1 >> 2),
            bsf(i2), bsf(i2 >> 2), bsf(i3), bsf(i3 >> 2)
        );

        _mm256_store_x1_epi32(y_ptr, y);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        s0 = _mm256_minwise2_pd(x0);
        s1 = _mm256_minwise2_pd(x1);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s1));

        __m128i y = _mm_setr_epi32(
            bsf(i0), bsf(i0 >> 2), bsf(i1), bsf(i1 >> 2)
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_load_x1_pd(x_ptr, x0);

        s0 = _mm256_minwise2_pd(x0);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));

        y_ptr[0] = bsf(i0);
        y_ptr[1] = bsf(i0 >> 2);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_maskload_x1_pd(x_ptr, x0, _mm256_setmask_pd(2));

        s0 = _mm256_minwise2_pd(x0);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));

        y_ptr[0] = bsf(i0);
    }

    return SUCCESS;
}

int ag_argmin_samples3_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if (samples != 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask3 = _mm256_setmask_pd(3);

    __m256d x0, x1, x2, x3, s0, s1, s2, s3;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE / 2) {
        x0 = _mm256_maskload_pd(x_ptr, mask3);
        x1 = _mm256_maskload_pd(x_ptr + 3, mask3);
        x2 = _mm256_maskload_pd(x_ptr + 6, mask3);
        x3 = _mm256_maskload_pd(x_ptr + 9, mask3);

        s0 = _mm256_minwise3_pd(x0);
        s1 = _mm256_minwise3_pd(x1);
        s2 = _mm256_minwise3_pd(x2);
        s3 = _mm256_minwise3_pd(x3);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s1));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s2));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s3));

        __m128i y = _mm_setr_epi32(
            bsf(i0), bsf(i1), bsf(i2), bsf(i3)
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += 12;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        x0 = _mm256_maskload_pd(x_ptr, mask3);
        x1 = _mm256_maskload_pd(x_ptr + 3, mask3);

        s0 = _mm256_minwise3_pd(x0);
        s1 = _mm256_minwise3_pd(x1);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s1));

        y_ptr[0] = bsf(i0);
        y_ptr[1] = bsf(i1);

        x_ptr += 6;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        x0 = _mm256_maskload_pd(x_ptr, mask3);

        s0 = _mm256_minwise3_pd(x0);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));

        y_ptr[0] = bsf(i0);
    }

    return SUCCESS;
}

int ag_argmin_samples4_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_DOUBLE_STRIDE) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, s0, s1, s2, s3;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        s0 = _mm256_minwise4_pd(x0);
        s1 = _mm256_minwise4_pd(x1);
        s2 = _mm256_minwise4_pd(x2);
        s3 = _mm256_minwise4_pd(x3);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s1));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s2));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s3));

        __m128i y = _mm_setr_epi32(
            bsf(i0), bsf(i1), bsf(i2), bsf(i3)
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        s0 = _mm256_minwise4_pd(x0);
        s1 = _mm256_minwise4_pd(x1);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s1));

        y_ptr[0] = bsf(i0);
        y_ptr[1] = bsf(i1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_load_x1_pd(x_ptr, x0);

        s0 = _mm256_minwise4_pd(x0);

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s0));

        y_ptr[0] = bsf(i0);
    }

    return SUCCESS;
}


int ag_argmin_samples5to7_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_DOUBLE_STRIDE || samples >= AVX2_DOUBLE_STRIDE * 2)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(samples & AVX2_DOUBLE_REMAIN_MASK);
    const __m256d pinf = _mm256_set1_pd(HUGE_VALF);

    __m256d x0, x1, x2, x3, x4, x5, x6, x7, s01, s23, s45, s67;
    uint i0, i1, i2, i3, i4, i5, i6, i7;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);
        _mm256_maskload_x2_pd(x_ptr + samples, x2, x3, mask);
        _mm256_maskload_x2_pd(x_ptr + samples * 2, x4, x5, mask);
        _mm256_maskload_x2_pd(x_ptr + samples * 3, x6, x7, mask);

        x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));
        x3 = _mm256_or_pd(x3, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));
        x5 = _mm256_or_pd(x5, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));
        x7 = _mm256_or_pd(x7, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));

        s01 = _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1));
        s23 = _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3));
        s45 = _mm256_min_pd(_mm256_minwise4_pd(x4), _mm256_minwise4_pd(x5));
        s67 = _mm256_min_pd(_mm256_minwise4_pd(x6), _mm256_minwise4_pd(x7));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s01));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s01));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s23));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s23));
        i4 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x4, s45));
        i5 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x5, s45));
        i6 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x6, s67));
        i7 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x7, s67));

        __m128i y = _mm_setr_epi32(
            bsf(i0 | (i1 << 4)), bsf(i2 | (i3 << 4)), bsf(i4 | (i5 << 4)), bsf(i6 | (i7 << 4))
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_DOUBLE_STRIDE * samples;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);
        _mm256_maskload_x2_pd(x_ptr + samples, x2, x3, mask);

        x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));
        x3 = _mm256_or_pd(x3, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));

        s01 = _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1));
        s23 = _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s01));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s01));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s23));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s23));

        y_ptr[0] = bsf(i0 | (i1 << 4));
        y_ptr[1] = bsf(i2 | (i3 << 4));

        x_ptr += AVX2_DOUBLE_STRIDE / 2 * samples;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

        x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));

        s01 = _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s01));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s01));

        y_ptr[0] = bsf(i0 | (i1 << 4));
    }

    return SUCCESS;
}

int ag_argmin_samples8_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_DOUBLE_STRIDE * 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, x4, x5, x6, x7, s01, s23, s45, s67;
    uint i0, i1, i2, i3, i4, i5, i6, i7;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_load_x8_pd(x_ptr, x0, x1, x2, x3, x4, x5, x6, x7);

        s01 = _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1));
        s23 = _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3));
        s45 = _mm256_min_pd(_mm256_minwise4_pd(x4), _mm256_minwise4_pd(x5));
        s67 = _mm256_min_pd(_mm256_minwise4_pd(x6), _mm256_minwise4_pd(x7));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s01));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s01));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s23));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s23));
        i4 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x4, s45));
        i5 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x5, s45));
        i6 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x6, s67));
        i7 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x7, s67));

        __m128i y = _mm_setr_epi32(
            bsf(i0 | (i1 << 4)), bsf(i2 | (i3 << 4)), bsf(i4 | (i5 << 4)), bsf(i6 | (i7 << 4))
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_DOUBLE_STRIDE * 8;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        s01 = _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1));
        s23 = _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s01));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s01));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s23));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s23));

        y_ptr[0] = bsf(i0 | (i1 << 4));
        y_ptr[1] = bsf(i2 | (i3 << 4));

        x_ptr += AVX2_DOUBLE_STRIDE / 2 * 8;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        s01 = _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s01));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s01));

        y_ptr[0] = bsf(i0 | (i1 << 4));
    }

    return SUCCESS;
}

int ag_argmin_samples9to11_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_DOUBLE_STRIDE * 2 || samples >= AVX2_DOUBLE_STRIDE * 3)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(samples & AVX2_DOUBLE_REMAIN_MASK);
    const __m256d pinf = _mm256_set1_pd(HUGE_VALF);

    __m256d x0, x1, x2, x3, x4, x5, s012, s345;
    uint i0, i1, i2, i3, i4, i5;

    uint r = n;

    while (r >= 2) {
        _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);
        _mm256_maskload_x3_pd(x_ptr + samples, x3, x4, x5, mask);

        x2 = _mm256_or_pd(x2, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));
        x5 = _mm256_or_pd(x5, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));

        s012 = _mm256_min_pd(_mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)), _mm256_minwise4_pd(x2));
        s345 = _mm256_min_pd(_mm256_min_pd(_mm256_minwise4_pd(x3), _mm256_minwise4_pd(x4)), _mm256_minwise4_pd(x5));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s012));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s012));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s012));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s345));
        i4 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x4, s345));
        i5 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x5, s345));

        y_ptr[0] = bsf(i0 | (i1 << 4) | (i2 << 8));
        y_ptr[1] = bsf(i3 | (i4 << 4) | (i5 << 8));

        x_ptr += 2 * samples;
        y_ptr += 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

        x2 = _mm256_or_pd(x2, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));

        s012 = _mm256_min_pd(_mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)), _mm256_minwise4_pd(x2));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s012));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s012));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s012));

        y_ptr[0] = bsf(i0 | (i1 << 4) | (i2 << 8));
    }

    return SUCCESS;
}

int ag_argmin_samples12_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_DOUBLE_STRIDE * 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, x4, x5, s012, s345;
    uint i0, i1, i2, i3, i4, i5;

    uint r = n;

    while (r >= 2) {
        _mm256_load_x3_pd(x_ptr, x0, x1, x2);
        _mm256_load_x3_pd(x_ptr + samples, x3, x4, x5);

        s012 = _mm256_min_pd(_mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)), _mm256_minwise4_pd(x2));
        s345 = _mm256_min_pd(_mm256_min_pd(_mm256_minwise4_pd(x3), _mm256_minwise4_pd(x4)), _mm256_minwise4_pd(x5));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s012));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s012));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s012));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s345));
        i4 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x4, s345));
        i5 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x5, s345));

        y_ptr[0] = bsf(i0 | (i1 << 4) | (i2 << 8));
        y_ptr[1] = bsf(i3 | (i4 << 4) | (i5 << 8));

        x_ptr += 2 * samples;
        y_ptr += 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_load_x3_pd(x_ptr, x0, x1, x2);

        s012 = _mm256_min_pd(_mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)), _mm256_minwise4_pd(x2));

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s012));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s012));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s012));

        y_ptr[0] = bsf(i0 | (i1 << 4) | (i2 << 8));
    }

    return SUCCESS;
}

int ag_argmin_samples13to15_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_DOUBLE_STRIDE * 3 || samples >= AVX2_DOUBLE_STRIDE * 4)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(samples & AVX2_DOUBLE_REMAIN_MASK);
    const __m256d pinf = _mm256_set1_pd(HUGE_VALF);

    __m256d x0, x1, x2, x3, s;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r > 0) {
        _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

        x3 = _mm256_or_pd(x3, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));

        s = _mm256_min_pd(
            _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)),
            _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3))
        );

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s));

        *y_ptr = bsf(i0 | (i1 << 4) | (i2 << 8) | (i3 << 12));

        x_ptr += samples;
        y_ptr++;
        r--;
    }

    return SUCCESS;
}

int ag_argmin_samples16_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_DOUBLE_STRIDE * 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3, s;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r > 0) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        s = _mm256_min_pd(
            _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)),
            _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3))
        );

        i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, s));
        i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, s));
        i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, s));
        i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, s));

        *y_ptr = bsf(i0 | (i1 << 4) | (i2 << 8) | (i3 << 12));

        x_ptr += samples;
        y_ptr++;
        r--;
    }

    return SUCCESS;
}

int ag_argmin_unaligned_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_DOUBLE_STRIDE * 4) || (samples % AVX2_DOUBLE_STRIDE == 0)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(samples & AVX2_DOUBLE_REMAIN_MASK);
    const __m256d pinf = _mm256_set1_pd(HUGE_VALF);

    __m256d x0, x1, x2, x3, s, t;
    uint i0, i1, i2, i3, index;

    for (uint i = 0; i < n; i++) {
        uint r = samples, k = 0;
        s = pinf;
        y_ptr[i] = 0;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);

            t = _mm256_min_pd(
                _mm256_min_pd(
                    _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)),
                    _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3))
                ),
                s
            );

            if (_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(s, t)) < 0xFu) {
                i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, t));
                i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, t));
                i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, t));
                i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, t));

                index = i0 | (i1 << 4) | (i2 << 8) | (i3 << 12);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
            k += AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(x_ptr, x0, x1);

            t = _mm256_min_pd(
                _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)),
                s
            );

            if (_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(s, t)) < 0xFu) {
                i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, t));
                i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, t));

                index = i0 | (i1 << 4);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
            k += AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(x_ptr, x0);

            t = _mm256_min_pd(_mm256_minwise4_pd(x0), s);

            if (_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(s, t)) < 0xFu) {
                i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, t));

                index = i0;

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
            k += AVX2_DOUBLE_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_pd(x_ptr, x0, mask);

            x0 = _mm256_or_pd(x0, _mm256_andnot_pd(_mm256_castsi256_pd(mask), pinf));

            t = _mm256_min_pd(_mm256_minwise4_pd(x0), s);

            if (_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(s, t)) < 0xFu) {
                i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, t));

                index = i0;

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += r;
        }
    }

    return SUCCESS;
}

int ag_argmin_aligned_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples % AVX2_DOUBLE_STRIDE != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d pinf = _mm256_set1_pd(HUGE_VALF);

    __m256d x0, x1, x2, x3, s, t;
    uint i0, i1, i2, i3, index;

    for (uint i = 0; i < n; i++) {
        uint r = samples, k = 0;
        s = pinf;
        y_ptr[i] = 0;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

            t = _mm256_min_pd(
                _mm256_min_pd(
                    _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)),
                    _mm256_min_pd(_mm256_minwise4_pd(x2), _mm256_minwise4_pd(x3))
                ),
                s
            );

            if (_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(s, t)) < 0xFu) {
                i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, t));
                i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, t));
                i2 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x2, t));
                i3 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x3, t));

                index = i0 | (i1 << 4) | (i2 << 8) | (i3 << 12);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
            k += AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(x_ptr, x0, x1);

            t = _mm256_min_pd(
                _mm256_min_pd(_mm256_minwise4_pd(x0), _mm256_minwise4_pd(x1)),
                s
            );

            if (_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(s, t)) < 0xFu) {
                i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, t));
                i1 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x1, t));

                index = i0 | (i1 << 4);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
            k += AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(x_ptr, x0);

            t = _mm256_min_pd(_mm256_minwise4_pd(x0), s);

            if (_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(s, t)) < 0xFu) {
                i0 = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, t));

                index = i0;

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_DOUBLE_STRIDE;
        }
    }

    return SUCCESS;
}

int ag_argmin_samplesleq8_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

    if (samples == 2) {
        return ag_argmin_samples2_d(n, samples, x_ptr, y_ptr);
    }
    else if (samples == 3) {
        return ag_argmin_samples3_d(n, samples, x_ptr, y_ptr);
    }
    else if (samples == 4) {
        return ag_argmin_samples4_d(n, samples, x_ptr, y_ptr);
    }
    else if (samples < AVX2_DOUBLE_STRIDE * 2) {
        return ag_argmin_samples5to7_d(n, samples, x_ptr, y_ptr);
    }
    else if (samples == AVX2_DOUBLE_STRIDE * 2) {
        return ag_argmin_samples8_d(n, samples, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int ag_argmin_samples_aligned_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

    if (samples == AVX2_DOUBLE_STRIDE) {
        return ag_argmin_samples4_d(n, samples, x_ptr, y_ptr);
    }
    if (samples == AVX2_DOUBLE_STRIDE * 2) {
        return ag_argmin_samples8_d(n, samples, x_ptr, y_ptr);
    }
    if (samples == AVX2_DOUBLE_STRIDE * 3) {
        return ag_argmin_samples12_d(n, samples, x_ptr, y_ptr);
    }
    if (samples == AVX2_DOUBLE_STRIDE * 4) {
        return ag_argmin_samples16_d(n, samples, x_ptr, y_ptr);
    }

    return ag_argmin_aligned_d(n, samples, x_ptr, y_ptr);
}

int ag_argmin_samples_unaligned_d(
    const uint n, const uint samples,
    indoubles x_ptr, outuints y_ptr) {

    if (samples <= AVX2_DOUBLE_STRIDE * 2) {
        return ag_argmin_samplesleq8_d(n, samples, x_ptr, y_ptr);
    }
    if (samples <= AVX2_DOUBLE_STRIDE * 3) {
        return ag_argmin_samples9to11_d(n, samples, x_ptr, y_ptr);
    }
    if (samples <= AVX2_DOUBLE_STRIDE * 4) {
        return ag_argmin_samples13to15_d(n, samples, x_ptr, y_ptr);
    }

    return ag_argmin_unaligned_d(n, samples, x_ptr, y_ptr);
}

#pragma managed

void AvxBlas::Aggregate::ArgMin(UInt32 n, UInt32 samples, UInt32 stride, Array<double>^ x, Array<Int32>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, samples, stride);

    Util::CheckLength(n * samples * stride, x);
    Util::CheckLength(n * stride, y);

    if (samples <= 1) {
        Initialize::Clear(n * stride, 0, y);
        return;
    }

    Array<double>^ xt;
    if (stride > 1) {
        xt = gcnew Array<double>(n * samples * stride, false);
        Permutate::Transpose(n, samples, stride, 1u, x, xt);
    }
    else {
        xt = x;
    }

    const double* x_ptr = (const double*)(xt->Ptr.ToPointer());
    uint* y_ptr = (uint*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((samples & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = ag_argmin_samples_aligned_d(n * stride, samples, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = ag_argmin_samples_unaligned_d(n * stride, samples, x_ptr, y_ptr);
    }

    if (stride > 1) {
        xt->~Array();
    }

    Util::AssertReturnCode(ret);
}
