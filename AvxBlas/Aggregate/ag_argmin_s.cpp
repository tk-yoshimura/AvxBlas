#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_min_s.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_cmp_s.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_loadstore_xn_epi32.hpp"
#include <memory.h>
#include <math.h>

using namespace System;

#pragma unmanaged

int ag_argmin_samples2_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, s0, s1, s2, s3;
    uint i0, i1, i2, i3;
    __m256i y01, y23;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE * 2) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        s0 = _mm256_minwise2_ps(x0);
        s1 = _mm256_minwise2_ps(x1);
        s2 = _mm256_minwise2_ps(x2);
        s3 = _mm256_minwise2_ps(x3);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s2));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s3));

        y01 = _mm256_setr_epi32(
            bsf(i0), bsf(i0 >> 2), bsf(i0 >> 4), bsf(i0 >> 6),
            bsf(i1), bsf(i1 >> 2), bsf(i1 >> 4), bsf(i1 >> 6)
        );

        y23 = _mm256_setr_epi32(
            bsf(i2), bsf(i2 >> 2), bsf(i2 >> 4), bsf(i2 >> 6),
            bsf(i3), bsf(i3 >> 2), bsf(i3 >> 4), bsf(i3 >> 6)
        );

        _mm256_store_x2_epi32(y_ptr, y01, y23);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_EPI32_STRIDE * 2;
        r -= AVX2_EPI32_STRIDE * 2;
    }
    if (r >= AVX2_EPI32_STRIDE) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        s0 = _mm256_minwise2_ps(x0);
        s1 = _mm256_minwise2_ps(x1);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));

        y01 = _mm256_setr_epi32(
            bsf(i0), bsf(i0 >> 2), bsf(i0 >> 4), bsf(i0 >> 6),
            bsf(i1), bsf(i1 >> 2), bsf(i1 >> 4), bsf(i1 >> 6)
        );

        _mm256_store_x1_epi32(y_ptr, y01);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_load_x1_ps(x_ptr, x0);

        s0 = _mm256_minwise2_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        __m128i y0 = _mm_setr_epi32(bsf(i0), bsf(i0 >> 2), bsf(i0 >> 4), bsf(i0 >> 6));

        _mm_store_epi32(y_ptr, y0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r > 0) {
        _mm256_maskload_x1_ps(x_ptr, x0, _mm256_setmask_ps(r * 2));

        s0 = _mm256_minwise2_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        __m128i y0 = _mm_setr_epi32(bsf(i0), bsf(i0 >> 2), bsf(i0 >> 4), bsf(i0 >> 6));

        _mm_maskstore_epi32((int*)y_ptr, _mm_setmask_ps(r), y0);
    }

    return SUCCESS;
}

int ag_argmin_samples3_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if (samples != 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask6 = _mm256_setmask_ps(6);

    __m256 x0, x1, x2, x3, s0, s1, s2, s3;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE) {
        x0 = _mm256_maskload_ps(x_ptr, mask6);
        x1 = _mm256_maskload_ps(x_ptr + 6, mask6);
        x2 = _mm256_maskload_ps(x_ptr + 12, mask6);
        x3 = _mm256_maskload_ps(x_ptr + 18, mask6);

        s0 = _mm256_minwise3_ps(x0);
        s1 = _mm256_minwise3_ps(x1);
        s2 = _mm256_minwise3_ps(x2);
        s3 = _mm256_minwise3_ps(x3);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s2));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s3));

        __m256i y = _mm256_setr_epi32(
            bsf(i0), bsf(i0 >> 3), bsf(i1), bsf(i1 >> 3),
            bsf(i2), bsf(i2 >> 3), bsf(i3), bsf(i3 >> 3)
        );

        _mm256_store_epi32(y_ptr, y);

        x_ptr += 24;
        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r >= AVX2_EPI32_STRIDE / 2) {
        x0 = _mm256_maskload_ps(x_ptr, mask6);
        x1 = _mm256_maskload_ps(x_ptr + 6, mask6);

        s0 = _mm256_minwise3_ps(x0);
        s1 = _mm256_minwise3_ps(x1);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));

        __m128i y = _mm_setr_epi32(bsf(i0), bsf(i0 >> 3), bsf(i1), bsf(i1 >> 3));

        _mm_store_epi32(y_ptr, y);

        x_ptr += 12;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        x0 = _mm256_maskload_ps(x_ptr, mask6);

        s0 = _mm256_minwise3_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        y_ptr[0] = bsf(i0);
        y_ptr[1] = bsf(i0 >> 3);

        x_ptr += 6;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        x0 = _mm256_maskload_ps(x_ptr, _mm256_setmask_ps(3));

        s0 = _mm256_minwise3_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        y_ptr[0] = bsf(i0);
    }

    return SUCCESS;
}

int ag_argmin_samples4_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, s0, s1, s2, s3;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        s0 = _mm256_minwise4_ps(x0);
        s1 = _mm256_minwise4_ps(x1);
        s2 = _mm256_minwise4_ps(x2);
        s3 = _mm256_minwise4_ps(x3);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s2));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s3));

        __m256i y = _mm256_setr_epi32(
            bsf(i0), bsf(i0 >> 4), bsf(i1), bsf(i1 >> 4),
            bsf(i2), bsf(i2 >> 4), bsf(i3), bsf(i3 >> 4)
        );

        _mm256_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        s0 = _mm256_minwise4_ps(x0);
        s1 = _mm256_minwise4_ps(x1);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));

        __m128i y = _mm_setr_epi32(bsf(i0), bsf(i0 >> 4), bsf(i1), bsf(i1 >> 4));

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_load_x1_ps(x_ptr, x0);

        s0 = _mm256_minwise4_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        y_ptr[0] = bsf(i0);
        y_ptr[1] = bsf(i0 >> 4);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        x0 = _mm256_maskload_ps(x_ptr, _mm256_setmask_ps(4));

        s0 = _mm256_minwise4_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        y_ptr[0] = bsf(i0);
    }

    return SUCCESS;
}

int ag_argmin_samples5to7_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_FLOAT_STRIDE / 2 || samples >= AVX2_FLOAT_STRIDE)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(samples);
    const __m256 pinf = _mm256_set1_ps(HUGE_VALF);

    __m256 x0, x1, x2, x3, x4, x5, x6, x7, s0, s1, s2, s3, s4, s5, s6, s7;
    uint i0, i1, i2, i3, i4, i5, i6, i7;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE) {
        x0 = _mm256_maskload_ps(x_ptr, mask);
        x1 = _mm256_maskload_ps(x_ptr + samples, mask);
        x2 = _mm256_maskload_ps(x_ptr + samples * 2, mask);
        x3 = _mm256_maskload_ps(x_ptr + samples * 3, mask);
        x4 = _mm256_maskload_ps(x_ptr + samples * 4, mask);
        x5 = _mm256_maskload_ps(x_ptr + samples * 5, mask);
        x6 = _mm256_maskload_ps(x_ptr + samples * 6, mask);
        x7 = _mm256_maskload_ps(x_ptr + samples * 7, mask);

        x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x4 = _mm256_or_ps(x4, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x5 = _mm256_or_ps(x5, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x6 = _mm256_or_ps(x6, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x7 = _mm256_or_ps(x7, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s0 = _mm256_minwise8_ps(x0);
        s1 = _mm256_minwise8_ps(x1);
        s2 = _mm256_minwise8_ps(x2);
        s3 = _mm256_minwise8_ps(x3);
        s4 = _mm256_minwise8_ps(x4);
        s5 = _mm256_minwise8_ps(x5);
        s6 = _mm256_minwise8_ps(x6);
        s7 = _mm256_minwise8_ps(x7);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s2));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s3));
        i4 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x4, s4));
        i5 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x5, s5));
        i6 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x6, s6));
        i7 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x7, s7));

        __m256i y = _mm256_setr_epi32(
            bsf(i0), bsf(i1), bsf(i2), bsf(i3), bsf(i4), bsf(i5), bsf(i6), bsf(i7)
        );

        _mm256_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE * samples;
        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r >= AVX2_EPI32_STRIDE / 2) {
        x0 = _mm256_maskload_ps(x_ptr, mask);
        x1 = _mm256_maskload_ps(x_ptr + samples, mask);
        x2 = _mm256_maskload_ps(x_ptr + samples * 2, mask);
        x3 = _mm256_maskload_ps(x_ptr + samples * 3, mask);

        x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s0 = _mm256_minwise8_ps(x0);
        s1 = _mm256_minwise8_ps(x1);
        s2 = _mm256_minwise8_ps(x2);
        s3 = _mm256_minwise8_ps(x3);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s2));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s3));

        __m128i y = _mm_setr_epi32(
            bsf(i0), bsf(i1), bsf(i2), bsf(i3)
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE / 2 * samples;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        x0 = _mm256_maskload_ps(x_ptr, mask);
        x1 = _mm256_maskload_ps(x_ptr + samples, mask);

        x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s0 = _mm256_minwise8_ps(x0);
        s1 = _mm256_minwise8_ps(x1);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));

        y_ptr[0] = bsf(i0);
        y_ptr[1] = bsf(i1);

        x_ptr += AVX2_FLOAT_STRIDE / 4 * samples;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s0 = _mm256_minwise8_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        y_ptr[0] = bsf(i0);
    }

    return SUCCESS;
}

int ag_argmin_samples8_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_FLOAT_STRIDE) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, x4, x5, x6, x7, s0, s1, s2, s3, s4, s5, s6, s7;
    uint i0, i1, i2, i3, i4, i5, i6, i7;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE) {
        _mm256_load_x8_ps(x_ptr, x0, x1, x2, x3, x4, x5, x6, x7);

        s0 = _mm256_minwise8_ps(x0);
        s1 = _mm256_minwise8_ps(x1);
        s2 = _mm256_minwise8_ps(x2);
        s3 = _mm256_minwise8_ps(x3);
        s4 = _mm256_minwise8_ps(x4);
        s5 = _mm256_minwise8_ps(x5);
        s6 = _mm256_minwise8_ps(x6);
        s7 = _mm256_minwise8_ps(x7);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s2));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s3));
        i4 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x4, s4));
        i5 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x5, s5));
        i6 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x6, s6));
        i7 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x7, s7));

        __m256i y = _mm256_setr_epi32(
            bsf(i0), bsf(i1), bsf(i2), bsf(i3), bsf(i4), bsf(i5), bsf(i6), bsf(i7)
        );

        _mm256_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE * 8;
        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        s0 = _mm256_minwise8_ps(x0);
        s1 = _mm256_minwise8_ps(x1);
        s2 = _mm256_minwise8_ps(x2);
        s3 = _mm256_minwise8_ps(x3);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s2));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s3));

        __m128i y = _mm_setr_epi32(
            bsf(i0), bsf(i1), bsf(i2), bsf(i3)
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE / 2 * 8;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        s0 = _mm256_minwise8_ps(x0);
        s1 = _mm256_minwise8_ps(x1);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s1));

        y_ptr[0] = bsf(i0);
        y_ptr[1] = bsf(i1);

        x_ptr += AVX2_FLOAT_STRIDE / 4 * 8;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_load_x1_ps(x_ptr, x0);

        s0 = _mm256_minwise8_ps(x0);

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s0));

        y_ptr[0] = bsf(i0);
    }

    return SUCCESS;
}

int ag_argmin_samples9to15_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_FLOAT_STRIDE || samples >= AVX2_FLOAT_STRIDE * 2)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(samples & AVX2_FLOAT_REMAIN_MASK);
    const __m256 pinf = _mm256_set1_ps(HUGE_VALF);

    __m256 x0, x1, x2, x3, x4, x5, x6, x7, s01, s23, s45, s67;
    uint i0, i1, i2, i3, i4, i5, i6, i7;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
        _mm256_maskload_x2_ps(x_ptr + samples, x2, x3, mask);
        _mm256_maskload_x2_ps(x_ptr + samples * 2, x4, x5, mask);
        _mm256_maskload_x2_ps(x_ptr + samples * 3, x6, x7, mask);

        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x5 = _mm256_or_ps(x5, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x7 = _mm256_or_ps(x7, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s01 = _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1));
        s23 = _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3));
        s45 = _mm256_min_ps(_mm256_minwise8_ps(x4), _mm256_minwise8_ps(x5));
        s67 = _mm256_min_ps(_mm256_minwise8_ps(x6), _mm256_minwise8_ps(x7));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s01));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s01));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s23));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s23));
        i4 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x4, s45));
        i5 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x5, s45));
        i6 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x6, s67));
        i7 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x7, s67));

        __m128i y = _mm_setr_epi32(
            bsf(i0 | (i1 << 8)), bsf(i2 | (i3 << 8)), bsf(i4 | (i5 << 8)), bsf(i6 | (i7 << 8))
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE / 2 * samples;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
        _mm256_maskload_x2_ps(x_ptr + samples, x2, x3, mask);

        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s01 = _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1));
        s23 = _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s01));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s01));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s23));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s23));

        y_ptr[0] = bsf(i0 | (i1 << 8));
        y_ptr[1] = bsf(i2 | (i3 << 8));

        x_ptr += AVX2_FLOAT_STRIDE / 4 * samples;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);

        x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s01 = _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s01));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s01));

        y_ptr[0] = bsf(i0 | (i1 << 8));
    }

    return SUCCESS;
}

int ag_argmin_samples16_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_FLOAT_STRIDE * 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, x4, x5, x6, x7, s01, s23, s45, s67;
    uint i0, i1, i2, i3, i4, i5, i6, i7;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE / 2) {
        _mm256_load_x8_ps(x_ptr, x0, x1, x2, x3, x4, x5, x6, x7);

        s01 = _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1));
        s23 = _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3));
        s45 = _mm256_min_ps(_mm256_minwise8_ps(x4), _mm256_minwise8_ps(x5));
        s67 = _mm256_min_ps(_mm256_minwise8_ps(x6), _mm256_minwise8_ps(x7));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s01));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s01));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s23));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s23));
        i4 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x4, s45));
        i5 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x5, s45));
        i6 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x6, s67));
        i7 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x7, s67));

        __m128i y = _mm_setr_epi32(
            bsf(i0 | (i1 << 8)), bsf(i2 | (i3 << 8)), bsf(i4 | (i5 << 8)), bsf(i6 | (i7 << 8))
        );

        _mm_store_epi32(y_ptr, y);

        x_ptr += AVX2_FLOAT_STRIDE / 2 * 16;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r >= AVX2_EPI32_STRIDE / 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        s01 = _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1));
        s23 = _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s01));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s01));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s23));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s23));

        y_ptr[0] = bsf(i0 | (i1 << 8));
        y_ptr[1] = bsf(i2 | (i3 << 8));

        x_ptr += AVX2_FLOAT_STRIDE / 4 * 16;
        y_ptr += AVX2_EPI32_STRIDE / 4;
        r -= AVX2_EPI32_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        s01 = _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s01));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s01));

        y_ptr[0] = bsf(i0 | (i1 << 8));
    }

    return SUCCESS;
}

int ag_argmin_samples17to23_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_FLOAT_STRIDE * 2 || samples >= AVX2_FLOAT_STRIDE * 3)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(samples & AVX2_FLOAT_REMAIN_MASK);
    const __m256 pinf = _mm256_set1_ps(HUGE_VALF);

    __m256 x0, x1, x2, x3, x4, x5, s012, s345;
    uint i0, i1, i2, i3, i4, i5;

    uint r = n;

    while (r >= 2) {
        _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);
        _mm256_maskload_x3_ps(x_ptr + samples, x3, x4, x5, mask);

        x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));
        x5 = _mm256_or_ps(x5, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s012 = _mm256_min_ps(_mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)), _mm256_minwise8_ps(x2));
        s345 = _mm256_min_ps(_mm256_min_ps(_mm256_minwise8_ps(x3), _mm256_minwise8_ps(x4)), _mm256_minwise8_ps(x5));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s012));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s012));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s012));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s345));
        i4 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x4, s345));
        i5 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x5, s345));

        y_ptr[0] = bsf(i0 | (i1 << 8) | (i2 << 16));
        y_ptr[1] = bsf(i3 | (i4 << 8) | (i5 << 16));

        x_ptr += 2 * samples;
        y_ptr += 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);

        x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s012 = _mm256_min_ps(_mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)), _mm256_minwise8_ps(x2));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s012));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s012));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s012));

        y_ptr[0] = bsf(i0 | (i1 << 8) | (i2 << 16));
    }

    return SUCCESS;
}

int ag_argmin_samples24_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_FLOAT_STRIDE * 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, x4, x5, s012, s345;
    uint i0, i1, i2, i3, i4, i5;

    uint r = n;

    while (r >= 2) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);
        _mm256_load_x3_ps(x_ptr + samples, x3, x4, x5);

        s012 = _mm256_min_ps(_mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)), _mm256_minwise8_ps(x2));
        s345 = _mm256_min_ps(_mm256_min_ps(_mm256_minwise8_ps(x3), _mm256_minwise8_ps(x4)), _mm256_minwise8_ps(x5));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s012));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s012));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s012));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s345));
        i4 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x4, s345));
        i5 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x5, s345));

        y_ptr[0] = bsf(i0 | (i1 << 8) | (i2 << 16));
        y_ptr[1] = bsf(i3 | (i4 << 8) | (i5 << 16));

        x_ptr += 2 * samples;
        y_ptr += 2;
        r -= 2;
    }
    if (r > 0) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);

        s012 = _mm256_min_ps(_mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)), _mm256_minwise8_ps(x2));

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s012));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s012));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s012));

        y_ptr[0] = bsf(i0 | (i1 << 8) | (i2 << 16));
    }

    return SUCCESS;
}

int ag_argmin_samples25to31_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_FLOAT_STRIDE * 3 || samples >= AVX2_FLOAT_STRIDE * 4)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(samples & AVX2_FLOAT_REMAIN_MASK);
    const __m256 pinf = _mm256_set1_ps(HUGE_VALF);

    __m256 x0, x1, x2, x3, s;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r > 0) {
        _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);

        x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

        s = _mm256_min_ps(
            _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)),
            _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3))
        );

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s));

        *y_ptr = bsf(i0 | (i1 << 8) | (i2 << 16) | (i3 << 24));

        x_ptr += samples;
        y_ptr++;
        r--;
    }

    return SUCCESS;
}

int ag_argmin_samples32_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples != AVX2_FLOAT_STRIDE * 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, s;
    uint i0, i1, i2, i3;

    uint r = n;

    while (r > 0) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        s = _mm256_min_ps(
            _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)),
            _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3))
        );

        i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, s));
        i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, s));
        i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, s));
        i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, s));

        *y_ptr = bsf(i0 | (i1 << 8) | (i2 << 16) | (i3 << 24));

        x_ptr += samples;
        y_ptr++;
        r--;
    }

    return SUCCESS;
}

int ag_argmin_unaligned_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples <= AVX2_FLOAT_STRIDE * 4) || (samples % AVX2_FLOAT_STRIDE == 0)) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(samples & AVX2_FLOAT_REMAIN_MASK);
    const __m256 pinf = _mm256_set1_ps(HUGE_VALF);

    __m256 x0, x1, x2, x3, s, t;
    uint i0, i1, i2, i3, index;

    for (uint i = 0; i < n; i++) {
        uint r = samples, k = 0;
        s = pinf;
        y_ptr[i] = 0;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);

            t = _mm256_min_ps(
                _mm256_min_ps(
                    _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)),
                    _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3))
                ),
                s
            );

            if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(s, t)) < 0xFFu) {
                i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, t));
                i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, t));
                i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, t));
                i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, t));

                index = i0 | (i1 << 8) | (i2 << 16) | (i3 << 24);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
            k += AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(x_ptr, x0, x1);

            t = _mm256_min_ps(
                _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)),
                s
            );

            if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(s, t)) < 0xFFu) {
                i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, t));
                i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, t));

                index = i0 | (i1 << 8);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
            k += AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(x_ptr, x0);

            t = _mm256_min_ps(_mm256_minwise8_ps(x0), s);

            if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(s, t)) < 0xFFu) {
                i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, t));

                index = i0;

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
            k += AVX2_FLOAT_STRIDE;
        }
        if (r > 0) {
            _mm256_maskload_x1_ps(x_ptr, x0, mask);

            x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), pinf));

            t = _mm256_min_ps(_mm256_minwise8_ps(x0), s);

            if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(s, t)) < 0xFFu) {
                i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, t));

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

int ag_argmin_aligned_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

#ifdef _DEBUG
    if ((samples % AVX2_FLOAT_STRIDE != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 pinf = _mm256_set1_ps(HUGE_VALF);

    __m256 x0, x1, x2, x3, s, t;
    uint i0, i1, i2, i3, index;

    for (uint i = 0; i < n; i++) {
        uint r = samples, k = 0;
        s = pinf;
        y_ptr[i] = 0;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

            t = _mm256_min_ps(
                _mm256_min_ps(
                    _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)),
                    _mm256_min_ps(_mm256_minwise8_ps(x2), _mm256_minwise8_ps(x3))
                ),
                s
            );

            if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(s, t)) < 0xFFu) {
                i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, t));
                i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, t));
                i2 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x2, t));
                i3 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x3, t));

                index = i0 | (i1 << 8) | (i2 << 16) | (i3 << 24);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
            k += AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(x_ptr, x0, x1);

            t = _mm256_min_ps(
                _mm256_min_ps(_mm256_minwise8_ps(x0), _mm256_minwise8_ps(x1)),
                s
            );

            if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(s, t)) < 0xFFu) {
                i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, t));
                i1 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x1, t));

                index = i0 | (i1 << 8);

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
            k += AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(x_ptr, x0);

            t = _mm256_min_ps(_mm256_minwise8_ps(x0), s);

            if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(s, t)) < 0xFFu) {
                i0 = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, t));

                index = i0;

                if (index > 0u) {
                    y_ptr[i] = bsf(index) + k;
                }

                s = t;
            }

            x_ptr += AVX2_FLOAT_STRIDE;
        }
    }

    return SUCCESS;
}

int ag_argmin_samplesleq8_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

    if (samples == 2) {
        return ag_argmin_samples2_s(n, samples, x_ptr, y_ptr);
    }
    else if (samples == 3) {
        return ag_argmin_samples3_s(n, samples, x_ptr, y_ptr);
    }
    else if (samples == 4) {
        return ag_argmin_samples4_s(n, samples, x_ptr, y_ptr);
    }
    else if (samples < AVX2_FLOAT_STRIDE) {
        return ag_argmin_samples5to7_s(n, samples, x_ptr, y_ptr);
    }
    else if (samples == AVX2_FLOAT_STRIDE) {
        return ag_argmin_samples8_s(n, samples, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int ag_argmin_samples_aligned_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

    if (samples == AVX2_FLOAT_STRIDE) {
        return ag_argmin_samples8_s(n, samples, x_ptr, y_ptr);
    }
    if (samples == AVX2_FLOAT_STRIDE * 2) {
        return ag_argmin_samples16_s(n, samples, x_ptr, y_ptr);
    }
    if (samples == AVX2_FLOAT_STRIDE * 3) {
        return ag_argmin_samples24_s(n, samples, x_ptr, y_ptr);
    }
    if (samples == AVX2_FLOAT_STRIDE * 4) {
        return ag_argmin_samples32_s(n, samples, x_ptr, y_ptr);
    }

    return ag_argmin_aligned_s(n, samples, x_ptr, y_ptr);
}

int ag_argmin_samples_unaligned_s(
    const uint n, const uint samples,
    infloats x_ptr, outuints y_ptr) {

    if (samples <= AVX2_FLOAT_STRIDE) {
        return ag_argmin_samplesleq8_s(n, samples, x_ptr, y_ptr);
    }
    if (samples <= AVX2_FLOAT_STRIDE * 2) {
        return ag_argmin_samples9to15_s(n, samples, x_ptr, y_ptr);
    }
    if (samples <= AVX2_FLOAT_STRIDE * 3) {
        return ag_argmin_samples17to23_s(n, samples, x_ptr, y_ptr);
    }
    if (samples <= AVX2_FLOAT_STRIDE * 4) {
        return ag_argmin_samples25to31_s(n, samples, x_ptr, y_ptr);
    }

    return ag_argmin_unaligned_s(n, samples, x_ptr, y_ptr);
}

#pragma managed

void AvxBlas::Aggregate::ArgMin(UInt32 n, UInt32 samples, UInt32 stride, Array<float>^ x, Array<Int32>^ y) {
    if (n <= 0 || samples <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, samples, stride);

    Util::CheckLength(n * samples * stride, x);
    Util::CheckLength(n * stride, y);

    if (samples == 1) {
        Initialize::Clear(n * stride, 0, y);
        return;
    }

    Array<float>^ xt;
    if (stride > 1) {
        xt = gcnew Array<float>(n * samples * stride, false);
        Permutate::Transpose(n, samples, stride, 1u, x, xt);
    }
    else {
        xt = x;
    }

    const float* x_ptr = (const float*)(xt->Ptr.ToPointer());
    uint* y_ptr = (uint*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((samples & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = ag_argmin_samples_aligned_s(n * stride, samples, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = ag_argmin_samples_unaligned_s(n * stride, samples, x_ptr, y_ptr);
    }

    if (stride > 1) {
        xt->~Array();
    }

    Util::AssertReturnCode(ret);
}
