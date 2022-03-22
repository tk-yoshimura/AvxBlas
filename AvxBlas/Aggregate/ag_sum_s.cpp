#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_sum_s.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include <memory.h>

using namespace System;

#pragma unmanaged

int ag_sum_stride1_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();
    const uint maskn = samples & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);

    for (uint i = 0; i < n; i++) {
        __m256 s = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_load_ps(x_ptr);

                s = _mm256_add_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                s = _mm256_add_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        if (r > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr, mask);

            s = _mm256_add_ps(x, s);

            x_ptr += r;
        }

        float y = _mm256_sum8to1_ps(s);

        *y_ptr = y;

        y_ptr += 1;
    }

    return SUCCESS;
}

int ag_sum_stride2_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const uint maskn = (2 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m128i mask2 = _mm_setmask_ps(2);

    for (uint i = 0; i < n; i++) {
        __m256 s = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                __m256 x = _mm256_load_ps(x_ptr);

                s = _mm256_add_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                s = _mm256_add_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        if (r > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr, mask);

            s = _mm256_add_ps(x, s);

            x_ptr += r * 2;
        }

        __m128 y = _mm256_sum8to2_ps(s);

        _mm_maskstore_ps(y_ptr, mask2, y);

        y_ptr += 2;
    }

    return SUCCESS;
}

int ag_sum_stride3_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const uint maskn = (3 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m128i mask3 = _mm_setmask_ps(3);

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero, s2 = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE) {
                __m256 x0, x1, x2;
                _mm256_load_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE) {
                __m256 x0, x1, x2;
                _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        if (r >= 6) { // 3 * r >= AVX2_FLOAT_STRIDE * 2
            __m256 x0, x1, x2;
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);
        }
        else if (r >= 3) { // 3 * r >= AVX2_FLOAT_STRIDE
            __m256 x0, x1;
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
        }
        else if (r >= 1) {
            __m256 x0;
            _mm256_maskload_x1_ps(x_ptr, x0, mask);

            s0 = _mm256_add_ps(x0, s0);
        }

        __m128 y = _mm256_sum24to3_ps(s0, s1, s2);

        _mm_maskstore_ps(y_ptr, mask3, y);

        x_ptr += 3 * r;
        y_ptr += 3;
    }

    return SUCCESS;
}

int ag_sum_stride4_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();
    const uint maskn = (4 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m256i mask4 = _mm256_setmask_ps(4);

    for (uint i = 0; i < n; i++) {
        __m256 s = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE / 4) {
                __m256 x = _mm256_load_ps(x_ptr);

                s = _mm256_add_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 4;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE / 4) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                s = _mm256_add_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 4;
            }
        }
        if (r > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr, mask4);

            s = _mm256_add_ps(x, s);

            x_ptr += AVX2_FLOAT_STRIDE / 2;
        }

        __m128 y = _mm256_sum8to4_ps(s);

        _mm_stream_ps(y_ptr, y);

        y_ptr += AVX2_FLOAT_STRIDE / 2;
    }

    return SUCCESS;
}

int ag_sum_stride5_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const uint maskn = (5 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m256i mask5 = _mm256_setmask_ps(5);

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero, s2 = zero, s3 = zero, s4 = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE) {
                __m256 x0, x1, x2, x3, x4;
                _mm256_load_x5_ps(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);
                s3 = _mm256_add_ps(x3, s3);
                s4 = _mm256_add_ps(x4, s4);

                x_ptr += AVX2_FLOAT_STRIDE * 5;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE) {
                __m256 x0, x1, x2, x3, x4;
                _mm256_loadu_x5_ps(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);
                s3 = _mm256_add_ps(x3, s3);
                s4 = _mm256_add_ps(x4, s4);

                x_ptr += AVX2_FLOAT_STRIDE * 5;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        if (r >= 7) { // 5 * r >= AVX2_FLOAT_STRIDE * 4
            __m256 x0, x1, x2, x3, x4;
            _mm256_maskload_x5_ps(x_ptr, x0, x1, x2, x3, x4, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);
            s3 = _mm256_add_ps(x3, s3);
            s4 = _mm256_add_ps(x4, s4);
        }
        else if (r >= 5) { // 5 * r >= AVX2_FLOAT_STRIDE * 3
            __m256 x0, x1, x2, x3;
            _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);
            s3 = _mm256_add_ps(x3, s3);
        }
        else if (r >= 4) { // 5 * r >= AVX2_FLOAT_STRIDE * 2
            __m256 x0, x1, x2;
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);
        }
        else if (r >= 2) { // 5 * r >= AVX2_FLOAT_STRIDE
            __m256 x0, x1;
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
        }
        else if (r >= 1) {
            __m256 x0;
            _mm256_maskload_x1_ps(x_ptr, x0, mask);

            s0 = _mm256_add_ps(x0, s0);
        }

        __m256 y = _mm256_sum40to5_ps(s0, s1, s2, s3, s4);

        _mm256_maskstore_ps(y_ptr, mask5, y);

        x_ptr += 5 * r;
        y_ptr += 5;
    }

    return SUCCESS;
}

int ag_sum_stride6_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const uint maskn = (6 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m256i mask6 = _mm256_setmask_ps(6);

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero, s2 = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                __m256 x0, x1, x2;
                _mm256_load_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                __m256 x0, x1, x2;
                _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        if (r >= 3) { // 6 * r >= AVX2_FLOAT_STRIDE * 2
            __m256 x0, x1, x2;
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);
        }
        else if (r >= 2) { // 6 * r >= AVX2_FLOAT_STRIDE
            __m256 x0, x1;
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
        }
        else if (r >= 1) {
            __m256 x0;
            _mm256_maskload_x1_ps(x_ptr, x0, mask);

            s0 = _mm256_add_ps(x0, s0);
        }

        __m256 y = _mm256_sum24to6_ps(s0, s1, s2);

        _mm256_maskstore_ps(y_ptr, mask6, y);

        x_ptr += 6 * r;
        y_ptr += 6;
    }

    return SUCCESS;
}

int ag_sum_stride7_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = _mm256_setmask_ps(7);

    for (uint i = 0; i < n; i++) {
        __m256 s = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x = _mm256_maskload_ps(x_ptr, mask);

            s = _mm256_add_ps(x, s);

            x_ptr += 7;
        }

        _mm256_maskstore_ps(y_ptr, mask, s);

        y_ptr += 7;
    }

    return SUCCESS;
}

int ag_sum_stride8_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (uint i = 0; i < n; i++) {
        __m256 s = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x = _mm256_load_ps(x_ptr);

            s = _mm256_add_ps(x, s);

            x_ptr += AVX2_FLOAT_STRIDE;
        }

        _mm256_stream_ps(y_ptr, s);

        y_ptr += AVX2_FLOAT_STRIDE;
    }

    return SUCCESS;
}

int ag_sum_stride9to15_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE || stride >= AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x0, x1;
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);

            x_ptr += stride;
        }

        _mm256_maskstore_x2_ps(y_ptr, s0, s1, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_sum_stride16_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x0, x1;
            _mm256_load_x2_ps(x_ptr, x0, x1);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);

            x_ptr += AVX2_FLOAT_STRIDE * 2;
        }

        _mm256_stream_x2_ps(y_ptr, s0, s1);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
    }

    return SUCCESS;
}

int ag_sum_stride17to23_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 2 || stride >= AVX2_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero, s2 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x0, x1, x2;
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);

            x_ptr += stride;
        }

        _mm256_maskstore_x3_ps(y_ptr, s0, s1, s2, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_sum_stride24_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero, s2 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x0, x1, x2;
            _mm256_load_x3_ps(x_ptr, x0, x1, x2);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);

            x_ptr += AVX2_FLOAT_STRIDE * 3;
        }

        _mm256_stream_x3_ps(y_ptr, s0, s1, s2);

        y_ptr += AVX2_FLOAT_STRIDE * 3;
    }

    return SUCCESS;
}

int ag_sum_stride25to31_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 3 || stride >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256 buf0 = zero, buf1 = zero, buf2 = zero, buf3 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x0, x1, x2, x3;
            _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);

            buf0 = _mm256_add_ps(x0, buf0);
            buf1 = _mm256_add_ps(x1, buf1);
            buf2 = _mm256_add_ps(x2, buf2);
            buf3 = _mm256_add_ps(x3, buf3);

            x_ptr += stride;
        }

        _mm256_maskstore_x4_ps(y_ptr, buf0, buf1, buf2, buf3, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_sum_stride32_s(
    const uint n, const uint samples,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (uint i = 0; i < n; i++) {
        __m256 s0 = zero, s1 = zero, s2 = zero, s3 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256 x0, x1, x2, x3;
            _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

            s0 = _mm256_add_ps(x0, s0);
            s1 = _mm256_add_ps(x1, s1);
            s2 = _mm256_add_ps(x2, s2);
            s3 = _mm256_add_ps(x3, s3);

            x_ptr += AVX2_FLOAT_STRIDE * 4;
        }

        _mm256_stream_x4_ps(y_ptr, s0, s1, s2, s3);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
    }

    return SUCCESS;
}

int ag_sum_strideleq8_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride == 1) {
        return ag_sum_stride1_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 2) {
        return ag_sum_stride2_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 3) {
        return ag_sum_stride3_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 4) {
        return ag_sum_stride4_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 5) {
        return ag_sum_stride5_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 6) {
        return ag_sum_stride6_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 7) {
        return ag_sum_stride7_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == AVX2_FLOAT_STRIDE) {
        return ag_sum_stride8_s(n, samples, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int ag_sum_aligned_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride == AVX2_FLOAT_STRIDE) {
        return ag_sum_stride8_s(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 2) {
        return ag_sum_stride16_s(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 3) {
        return ag_sum_stride24_s(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 4) {
        return ag_sum_stride32_s(n, samples, x_ptr, y_ptr);
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    float* s_ptr = (float*)_aligned_malloc((size_t)stride * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            _mm256_store_ps(s_ptr + c, zero);
        }

        for (uint j = 0; j < samples; j++) {
            for (uint c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_load_ps(x_ptr + c);
                __m256 s = _mm256_load_ps(s_ptr + c);

                s = _mm256_add_ps(x, s);

                _mm256_store_ps(s_ptr + c, s);
            }

            x_ptr += stride;
        }

        for (uint c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            __m256 s = _mm256_load_ps(s_ptr + c);

            _mm256_stream_ps(y_ptr + c, s);
        }

        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int ag_sum_unaligned_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride <= AVX2_FLOAT_STRIDE) {
        return ag_sum_strideleq8_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 2) {
        return ag_sum_stride9to15_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 3) {
        return ag_sum_stride17to23_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 4) {
        return ag_sum_stride25to31_s(n, samples, stride, x_ptr, y_ptr);
    }

    const uint sb = stride & AVX2_FLOAT_BATCH_MASK, sr = stride - sb;

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = _mm256_setmask_ps(sr);

    float* s_ptr = (float*)_aligned_malloc(((size_t)stride + AVX2_FLOAT_STRIDE) * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            _mm256_store_ps(s_ptr + c, zero);
        }

        for (uint j = 0; j < samples; j++) {
            for (uint c = 0; c < sb; c += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr + c);
                __m256 s = _mm256_load_ps(s_ptr + c);

                s = _mm256_add_ps(x, s);

                _mm256_store_ps(s_ptr + c, s);
            }
            if (sr > 0) {
                __m256 x = _mm256_maskload_ps(x_ptr + sb, mask);
                __m256 y = _mm256_load_ps(s_ptr + sb);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(s_ptr + sb, y);
            }

            x_ptr += stride;
        }

        for (uint c = 0; c < sb; c += AVX2_FLOAT_STRIDE) {
            __m256 y = _mm256_load_ps(s_ptr + c);

            _mm256_storeu_ps(y_ptr + c, y);
        }
        if (sr > 0) {
            __m256 y = _mm256_load_ps(s_ptr + sb);

            _mm256_maskstore_ps(y_ptr + sb, mask, y);
        }

        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int ag_sum_batch_s(
    const uint n, const uint g, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    const uint sg = stride * g;

#ifdef _DEBUG
    if ((sg & AVX2_FLOAT_REMAIN_MASK) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint sb = samples / g * g, sr = samples - sb;
    const uint rem = stride * sr;
    const uint remb = rem & AVX2_FLOAT_BATCH_MASK, remr = rem - remb;
    const __m256i mask = _mm256_setmask_ps(remr);

    const __m256 zero = _mm256_setzero_ps();

    float* buf = (float*)_aligned_malloc((size_t)sg * sizeof(float), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < sg; c += AVX2_FLOAT_STRIDE) {
            _mm256_store_ps(buf + c, zero);
        }

        for (uint s = 0; s < sb; s += g) {
            for (uint c = 0; c < sg; c += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr + c);
                __m256 y = _mm256_load_ps(buf + c);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(buf + c, y);
            }
            x_ptr += sg;
        }
        if (sr > 0) {
            for (uint c = 0; c < remb; c += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr + c);
                __m256 y = _mm256_load_ps(buf + c);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(buf + c, y);
            }
            if (remr > 0) {
                __m256 x = _mm256_maskload_ps(x_ptr + remb, mask);
                __m256 y = _mm256_load_ps(buf + remb);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(buf + remb, y);
            }
            x_ptr += rem;
        }

        ag_sum_unaligned_s(1, g, stride, buf, y_ptr);

        y_ptr += stride;
    }

    _aligned_free(buf);

    return SUCCESS;
}

#pragma managed

void AvxBlas::Aggregate::Sum(UInt32 n, UInt32 samples, UInt32 stride, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || samples <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, samples, stride);

    Util::CheckLength(n * samples * stride, x);
    Util::CheckLength(n * stride, y);

    if (samples == 1) {
        Elementwise::Copy(n * stride, x, y);
        return;
    }

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (stride <= 6u) {
#ifdef _DEBUG
        Console::WriteLine("type leq6");
#endif // _DEBUG

        ret = ag_sum_strideleq8_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = ag_sum_aligned_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride <= MAX_AGGREGATE_BATCHING) {
        UInt32 g = Numeric::LCM(stride, AVX2_FLOAT_STRIDE) / stride;

        if (samples >= g * 4) {
#ifdef _DEBUG
            Console::WriteLine("type batch g:" + g.ToString());
#endif // _DEBUG

            ret = ag_sum_batch_s(n, g, samples, stride, x_ptr, y_ptr);
        }
    }
    if (ret == UNEXECUTED) {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = ag_sum_unaligned_s(n, samples, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
