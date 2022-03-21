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
    const uint sb = samples & AVX2_DOUBLE_BATCH_MASK, sr = samples - sb;
    const __m256i mask = _mm256_setmask_pd(sr);

    if (sr > 0) {
        for (uint i = 0; i < n; i++) {
            __m256d buf = zero;

            for (uint s = 0; s < sb; s += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr);

                buf = _mm256_add_pd(x, buf);

                x_ptr += AVX2_DOUBLE_STRIDE;
            }
            {
                __m256d x = _mm256_maskload_pd(x_ptr, mask);

                buf = _mm256_add_pd(x, buf);

                x_ptr += sr;
            }

            double y = _mm256_sum4to1_pd(buf);

            *y_ptr = y;

            y_ptr += 1;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            __m256d buf = zero;

            for (uint s = 0; s < sb; s += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_load_pd(x_ptr);

                buf = _mm256_add_pd(x, buf);

                x_ptr += AVX2_DOUBLE_STRIDE;
            }

            double y = _mm256_sum4to1_pd(buf);

            *y_ptr = y;

            y_ptr += 1;
        }
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
    const uint sb = samples / 2 * 2, sr = samples - sb;
    const __m256i mask = _mm256_setmask_pd(sr * 2);

    if (sr > 0) {
        for (uint i = 0; i < n; i++) {
            __m256d buf = zero;

            for (uint s = 0; s < sb; s += 2) {
                __m256d x = _mm256_loadu_pd(x_ptr);

                buf = _mm256_add_pd(x, buf);

                x_ptr += AVX2_DOUBLE_STRIDE;
            }
            {
                __m256d x = _mm256_maskload_pd(x_ptr, mask);

                buf = _mm256_add_pd(x, buf);

                x_ptr += sr * 2;
            }

            __m128d y = _mm256_sum4to2_pd(buf);

            _mm_stream_pd(y_ptr, y);

            y_ptr += 2;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            __m256d buf = zero;

            for (uint s = 0; s < sb; s += 2) {
                __m256d x = _mm256_load_pd(x_ptr);

                buf = _mm256_add_pd(x, buf);

                x_ptr += AVX2_DOUBLE_STRIDE;
            }

            __m128d y = _mm256_sum4to2_pd(buf);

            _mm_stream_pd(y_ptr, y);

            y_ptr += 2;
        }
    }

    return SUCCESS;
}

int ag_sum_stride3_d(
    const uint n, const uint samples,
    indoubles x_ptr, outdoubles y_ptr) {

    const __m256d zero = _mm256_setzero_pd();
    const __m256i mask = _mm256_setmask_pd(3);

    for (uint i = 0; i < n; i++) {
        __m256d buf = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x = _mm256_maskload_pd(x_ptr, mask);

            buf = _mm256_add_pd(x, buf);

            x_ptr += 3;
        }

        _mm256_maskstore_pd(y_ptr, mask, buf);

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
        __m256d buf = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x = _mm256_load_pd(x_ptr);

            buf = _mm256_add_pd(x, buf);

            x_ptr += AVX2_DOUBLE_STRIDE;
        }

        _mm256_stream_pd(y_ptr, buf);

        y_ptr += AVX2_DOUBLE_STRIDE;
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
        __m256d buf0 = zero, buf1 = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x0, x1;
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);

            x_ptr += stride;
        }

        _mm256_maskstore_x2_pd(y_ptr, buf0, buf1, mask);

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
        __m256d buf0 = zero, buf1 = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x0, x1;
            _mm256_load_x2_pd(x_ptr, x0, x1);

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
        }

        _mm256_stream_x2_pd(y_ptr, buf0, buf1);

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
        __m256d buf0 = zero, buf1 = zero, buf2 = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x0, x1, x2;
            _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);
            buf2 = _mm256_add_pd(x2, buf2);

            x_ptr += stride;
        }

        _mm256_maskstore_x3_pd(y_ptr, buf0, buf1, buf2, mask);

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
        __m256d buf0 = zero, buf1 = zero, buf2 = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x0, x1, x2;
            _mm256_load_x3_pd(x_ptr, x0, x1, x2);

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);
            buf2 = _mm256_add_pd(x2, buf2);

            x_ptr += AVX2_DOUBLE_STRIDE * 3;
        }

        _mm256_stream_x3_pd(y_ptr, buf0, buf1, buf2);

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
        __m256d buf0 = zero, buf1 = zero, buf2 = zero, buf3 = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x0, x1, x2, x3;
            _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);
            buf2 = _mm256_add_pd(x2, buf2);
            buf3 = _mm256_add_pd(x3, buf3);

            x_ptr += stride;
        }

        _mm256_maskstore_x4_pd(y_ptr, buf0, buf1, buf2, buf3, mask);

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
        __m256d buf0 = zero, buf1 = zero, buf2 = zero, buf3 = zero;

        for (uint s = 0; s < samples; s++) {
            __m256d x0, x1, x2, x3;
            _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);
            buf2 = _mm256_add_pd(x2, buf2);
            buf3 = _mm256_add_pd(x3, buf3);

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
        }

        _mm256_stream_x4_pd(y_ptr, buf0, buf1, buf2, buf3);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
    }

    return SUCCESS;
}

int ag_sum_strideleq4_d(
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
    else if (stride == AVX2_DOUBLE_STRIDE) {
        return ag_sum_stride4_d(n, samples, x_ptr, y_ptr);
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

    double* buf = (double*)_aligned_malloc((size_t)stride * sizeof(double), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(buf + c, zero);
        }

        for (uint s = 0; s < samples; s++) {
            for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_load_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(buf + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + c, y);
            }

            x_ptr += stride;
        }

        for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            __m256d y = _mm256_load_pd(buf + c);

            _mm256_stream_pd(y_ptr + c, y);
        }

        y_ptr += stride;
    }

    _aligned_free(buf);

    return SUCCESS;
}

int ag_sum_unaligned_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride <= AVX2_DOUBLE_STRIDE) {
        return ag_sum_strideleq4_d(n, samples, stride, x_ptr, y_ptr);
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

    double* buf = (double*)_aligned_malloc(((size_t)stride + AVX2_DOUBLE_STRIDE) * sizeof(double), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(buf + c, zero);
        }

        for (uint s = 0; s < samples; s++) {
            for (uint c = 0; c < sb; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(buf + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + c, y);
            }
            if (sr > 0) {
                __m256d x = _mm256_maskload_pd(x_ptr + sb, mask);
                __m256d y = _mm256_load_pd(buf + sb);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + sb, y);
            }

            x_ptr += stride;
        }

        for (uint c = 0; c < sb; c += AVX2_DOUBLE_STRIDE) {
            __m256d y = _mm256_load_pd(buf + c);

            _mm256_storeu_pd(y_ptr + c, y);
        }
        if (sr > 0) {
            __m256d y = _mm256_load_pd(buf + sb);

            _mm256_maskstore_pd(y_ptr + sb, mask, y);
        }

        y_ptr += stride;
    }

    _aligned_free(buf);

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

    double* buf = (double*)_aligned_malloc((size_t)sg * sizeof(double), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < sg; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(buf + c, zero);
        }

        for (uint s = 0; s < sb; s += g) {
            for (uint c = 0; c < sg; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(buf + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + c, y);
            }
            x_ptr += sg;
        }
        if (sr > 0) {
            for (uint c = 0; c < remb; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(buf + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + c, y);
            }
            if (remr > 0) {
                __m256d x = _mm256_maskload_pd(x_ptr + remb, mask);
                __m256d y = _mm256_load_pd(buf + remb);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + remb, y);
            }
            x_ptr += rem;
        }

        ag_sum_unaligned_d(1, g, stride, buf, y_ptr);

        y_ptr += stride;
    }

    _aligned_free(buf);

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

    if (stride == 1u) {
#ifdef _DEBUG
        Console::WriteLine("type stride1");
#endif // _DEBUG

        ret = ag_sum_stride1_d(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 2u) {
#ifdef _DEBUG
        Console::WriteLine("type stride2");
#endif // _DEBUG

        ret = ag_sum_stride2_d(n, samples, x_ptr, y_ptr);
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
