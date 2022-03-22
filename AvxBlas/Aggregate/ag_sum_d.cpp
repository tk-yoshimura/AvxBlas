#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_sum_d.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"
#include <memory.h>

using namespace System;

#pragma unmanaged

int ag_sum_stride1_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

    const __m256d zero = _mm256_setzero_pd();
    const uint maskn = samples & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    for (uint i = 0; i < n; i++) {
        __m256d s = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_load_pd(x_ptr);

                s = _mm256_add_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr);

                s = _mm256_add_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        if (r > 0) {
            __m256d x = _mm256_maskload_pd(x_ptr, mask);

            s = _mm256_add_pd(x, s);

            x_ptr += r;
        }

        double y = _mm256_sum4to1_pd(s);

        *y_ptr = y;

        y_ptr += 1;
    }

    return SUCCESS;
}

int ag_sum_stride2_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();
    const uint maskn = (2 * samples) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    for (uint i = 0; i < n; i++) {
        __m256d s = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE / 2) {
                __m256d x = _mm256_load_pd(x_ptr);

                s = _mm256_add_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE / 2;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE / 2) {
                __m256d x = _mm256_loadu_pd(x_ptr);

                s = _mm256_add_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE / 2;
            }
        }
        if (r > 0) {
            __m256d x = _mm256_maskload_pd(x_ptr, mask);

            s = _mm256_add_pd(x, s);

            x_ptr += r * 2;
        }

        __m128d y = _mm256_sum4to2_pd(s);

        _mm_store_pd(y_ptr, y);

        y_ptr += 2;
    }

    return SUCCESS;
}

int ag_sum_stride3_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

    const __m256d zero = _mm256_setzero_pd();

    const __m256i mask = _mm256_setmask_pd((3 * samples) & AVX2_DOUBLE_REMAIN_MASK);
    const __m256i mask3 = _mm256_setmask_pd(3);

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero, s2 = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE) {
                __m256d x0, x1, x2;
                _mm256_load_x3_pd(x_ptr, x0, x1, x2);

                s0 = _mm256_add_pd(x0, s0);
                s1 = _mm256_add_pd(x1, s1);
                s2 = _mm256_add_pd(x2, s2);

                x_ptr += AVX2_DOUBLE_STRIDE * 3;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE) {
                __m256d x0, x1, x2;
                _mm256_loadu_x3_pd(x_ptr, x0, x1, x2);

                s0 = _mm256_add_pd(x0, s0);
                s1 = _mm256_add_pd(x1, s1);
                s2 = _mm256_add_pd(x2, s2);

                x_ptr += AVX2_DOUBLE_STRIDE * 3;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        if (r >= 3) { // 3 * r >= AVX2_DOUBLE_STRIDE * 2
            __m256d x0, x1, x2;
            _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
            s2 = _mm256_add_pd(x2, s2);
        }
        else if (r >= 2) { // 3 * r >= AVX2_DOUBLE_STRIDE
            __m256d x0, x1;
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
        }
        else if (r >= 1) {
            __m256d x0;
            _mm256_maskload_x1_pd(x_ptr, x0, mask);

            s0 = _mm256_add_pd(x0, s0);
        }

        __m256d y = _mm256_sum12to3_pd(s0, s1, s2);

        _mm256_maskstore_pd(y_ptr, mask3, y);

        x_ptr += 3 * r;
        y_ptr += 3;
    }

    return SUCCESS;
}

int ag_sum_stride4_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (uint i = 0; i < n; i++) {
        __m256d s = zero;

        for (uint j = 0; j < samples; j++) {
            __m256d x = _mm256_load_pd(x_ptr);

            s = _mm256_add_pd(x, s);

            x_ptr += AVX2_DOUBLE_STRIDE;
        }

        _mm256_stream_pd(y_ptr, s);

        y_ptr += AVX2_DOUBLE_STRIDE;
    }

    return SUCCESS;
}

int ag_sum_stride5_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

    const __m256d zero = _mm256_setzero_pd();
    const uint maskn = (5 * samples) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);
    const __m256i mask1 = _mm256_setmask_pd(1);

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero, s2 = zero, s3 = zero, s4 = zero;
        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE) {
                __m256d x0, x1, x2, x3, x4;
                _mm256_load_x5_pd(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_add_pd(x0, s0);
                s1 = _mm256_add_pd(x1, s1);
                s2 = _mm256_add_pd(x2, s2);
                s3 = _mm256_add_pd(x3, s3);
                s4 = _mm256_add_pd(x4, s4);

                x_ptr += AVX2_DOUBLE_STRIDE * 5;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE) {
                __m256d x0, x1, x2, x3, x4;
                _mm256_loadu_x5_pd(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_add_pd(x0, s0);
                s1 = _mm256_add_pd(x1, s1);
                s2 = _mm256_add_pd(x2, s2);
                s3 = _mm256_add_pd(x3, s3);
                s4 = _mm256_add_pd(x4, s4);

                x_ptr += AVX2_DOUBLE_STRIDE * 5;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        if (r >= 3) { // 5 * r >= AVX2_DOUBLE_STRIDE * 3
            __m256d x0, x1, x2, x3;
            _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
            s2 = _mm256_add_pd(x2, s2);
            s3 = _mm256_add_pd(x3, s3);
        }
        else if (r >= 2) { // 5 * r >= AVX2_DOUBLE_STRIDE * 2
            __m256d x0, x1, x2;
            _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
            s2 = _mm256_add_pd(x2, s2);
        }
        else if (r >= 1) { // 5 * r >= AVX2_DOUBLE_STRIDE
            __m256d x0, x1;
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
        }

        __m256dx2 y = _mm256_sum20to5_pd(s0, s1, s2, s3, s4);

        _mm256_maskstore_x2_pd(y_ptr, y.lo, y.hi, mask1);

        x_ptr += 5 * r;
        y_ptr += 5;
    }

    return SUCCESS;
}

int ag_sum_stride5to7_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE || stride >= AVX2_DOUBLE_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();
    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256d x0, x1;
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);

            x_ptr += stride;
        }

        _mm256_maskstore_x2_pd(y_ptr, s0, s1, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_sum_stride8_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256d x0, x1;
            _mm256_load_x2_pd(x_ptr, x0, x1);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
        }

        _mm256_stream_x2_pd(y_ptr, s0, s1);

        y_ptr += AVX2_DOUBLE_STRIDE * 2;
    }

    return SUCCESS;
}

int ag_sum_stride9to11_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 2 || stride >= AVX2_DOUBLE_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();
    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero, s2 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256d x0, x1, x2;
            _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
            s2 = _mm256_add_pd(x2, s2);

            x_ptr += stride;
        }

        _mm256_maskstore_x3_pd(y_ptr, s0, s1, s2, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_sum_stride12_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero, s2 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256d x0, x1, x2;
            _mm256_load_x3_pd(x_ptr, x0, x1, x2);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
            s2 = _mm256_add_pd(x2, s2);

            x_ptr += AVX2_DOUBLE_STRIDE * 3;
        }

        _mm256_stream_x3_pd(y_ptr, s0, s1, s2);

        y_ptr += AVX2_DOUBLE_STRIDE * 3;
    }

    return SUCCESS;
}

int ag_sum_stride13to15_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 3 || stride >= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();
    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero, s2 = zero, s3 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256d x0, x1, x2, x3;
            _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
            s2 = _mm256_add_pd(x2, s2);
            s3 = _mm256_add_pd(x3, s3);

            x_ptr += stride;
        }

        _mm256_maskstore_x4_pd(y_ptr, s0, s1, s2, s3, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_sum_stride16_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (uint i = 0; i < n; i++) {
        __m256d s0 = zero, s1 = zero, s2 = zero, s3 = zero;

        for (uint j = 0; j < samples; j++) {
            __m256d x0, x1, x2, x3;
            _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

            s0 = _mm256_add_pd(x0, s0);
            s1 = _mm256_add_pd(x1, s1);
            s2 = _mm256_add_pd(x2, s2);
            s3 = _mm256_add_pd(x3, s3);

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
        }

        _mm256_stream_x4_pd(y_ptr, s0, s1, s2, s3);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
    }

    return SUCCESS;
}

int ag_sum_strideleq5_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride == 1) {
        return ag_sum_stride1_d(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 2) {
        return ag_sum_stride2_d(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 3) {
        return ag_sum_stride3_d(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 4) {
        return ag_sum_stride4_d(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 5) {
        return ag_sum_stride5_d(n, samples, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int ag_sum_aligned_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride == AVX2_DOUBLE_STRIDE) {
        return ag_sum_stride4_d(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 2) {
        return ag_sum_stride8_d(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 3) {
        return ag_sum_stride12_d(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 4) {
        return ag_sum_stride16_d(n, samples, x_ptr, y_ptr);
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    double* s_ptr = (double*)_aligned_malloc((size_t)stride * sizeof(double), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(s_ptr + c, zero);
        }

        for (uint j = 0; j < samples; j++) {
            for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_load_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(s_ptr + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(s_ptr + c, y);
            }

            x_ptr += stride;
        }

        for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            __m256d y = _mm256_load_pd(s_ptr + c);

            _mm256_stream_pd(y_ptr + c, y);
        }

        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int ag_sum_unaligned_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride <= 5) {
        return ag_sum_strideleq5_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 2) {
        return ag_sum_stride5to7_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 3) {
        return ag_sum_stride9to11_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 4) {
        return ag_sum_stride13to15_d(n, samples, stride, x_ptr, y_ptr);
    }

    const uint sb = stride & AVX2_DOUBLE_BATCH_MASK, sr = stride - sb;

    const __m256d zero = _mm256_setzero_pd();
    const __m256i mask = _mm256_setmask_pd(sr);

    double* s_ptr = (double*)_aligned_malloc(((size_t)stride + AVX2_DOUBLE_STRIDE) * sizeof(double), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(s_ptr + c, zero);
        }

        for (uint j = 0; j < samples; j++) {
            for (uint c = 0; c < sb; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(s_ptr + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(s_ptr + c, y);
            }
            if (sr > 0) {
                __m256d x = _mm256_maskload_pd(x_ptr + sb, mask);
                __m256d y = _mm256_load_pd(s_ptr + sb);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(s_ptr + sb, y);
            }

            x_ptr += stride;
        }

        for (uint c = 0; c < sb; c += AVX2_DOUBLE_STRIDE) {
            __m256d y = _mm256_load_pd(s_ptr + c);

            _mm256_storeu_pd(y_ptr + c, y);
        }
        if (sr > 0) {
            __m256d y = _mm256_load_pd(s_ptr + sb);

            _mm256_maskstore_pd(y_ptr + sb, mask, y);
        }

        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int ag_sum_batch_d(
    const uint n, const uint g, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    const uint sg = stride * g;

#ifdef _DEBUG
    if ((sg & AVX2_DOUBLE_REMAIN_MASK) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint sb = samples / g * g, sr = samples - sb;
    const uint rem = stride * sr;
    const uint remb = rem & AVX2_DOUBLE_BATCH_MASK, remr = rem - remb;
    const __m256i mask = _mm256_setmask_pd(remr);

    const __m256d zero = _mm256_setzero_pd();

    double* s_ptr = (double*)_aligned_malloc((size_t)sg * sizeof(double), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < sg; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(s_ptr + c, zero);
        }

        for (uint s = 0; s < sb; s += g) {
            for (uint c = 0; c < sg; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(s_ptr + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(s_ptr + c, y);
            }
            x_ptr += sg;
        }
        if (sr > 0) {
            for (uint c = 0; c < remb; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(s_ptr + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(s_ptr + c, y);
            }
            if (remr > 0) {
                __m256d x = _mm256_maskload_pd(x_ptr + remb, mask);
                __m256d y = _mm256_load_pd(s_ptr + remb);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(s_ptr + remb, y);
            }
            x_ptr += rem;
        }

        ag_sum_unaligned_d(1, g, stride, s_ptr, y_ptr);

        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

#pragma managed

void AvxBlas::Aggregate::Sum(UInt32 n, UInt32 samples, UInt32 stride, Array<double>^ x, Array<double>^ y) {
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

    const double* x_ptr = (const double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (stride <= 5u) {
#ifdef _DEBUG
        Console::WriteLine("type strideleq5");
#endif // _DEBUG

        ret = ag_sum_strideleq5_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = ag_sum_aligned_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride <= MAX_AGGREGATE_BATCHING) {
        UInt32 g = Numeric::LCM(stride, AVX2_DOUBLE_STRIDE) / stride;

        if (samples >= g * 4) {
#ifdef _DEBUG
            Console::WriteLine("type batch g:" + g.ToString());
#endif // _DEBUG

            ret = ag_sum_batch_d(n, g, samples, stride, x_ptr, y_ptr);
        }
    }
    if (ret == UNEXECUTED) {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = ag_sum_unaligned_d(n, samples, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
