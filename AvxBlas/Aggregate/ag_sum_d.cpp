#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_sum_d.hpp"
#include <memory.h>

using namespace System;

#pragma unmanaged

int ag_sum_stride1_d(
    const unsigned int n, const unsigned int samples,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

    const __m256d zero = _mm256_setzero_pd();
    const unsigned int sb = samples & AVX2_DOUBLE_BATCH_MASK, sr = samples - sb;
    const __m256i mask = _mm256_setmask_pd(sr);

    if (sr > 0) {
        for (unsigned int i = 0; i < n; i++) {
            __m256d buf = zero;

            for (unsigned int s = 0; s < sb; s += AVX2_DOUBLE_STRIDE) {
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
        for (unsigned int i = 0; i < n; i++) {
            __m256d buf = zero;

            for (unsigned int s = 0; s < sb; s += AVX2_DOUBLE_STRIDE) {
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
    const unsigned int n, const unsigned int samples,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();
    const unsigned int sb = samples / 2 * 2, sr = samples - sb;
    const __m256i mask = _mm256_setmask_pd(sr * 2);

    if (sr > 0) {
        for (unsigned int i = 0; i < n; i++) {
            __m256d buf = zero;

            for (unsigned int s = 0; s < sb; s += 2) {
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
        for (unsigned int i = 0; i < n; i++) {
            __m256d buf = zero;

            for (unsigned int s = 0; s < sb; s += 2) {
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
    const unsigned int n, const unsigned int samples,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

    const __m256d zero = _mm256_setzero_pd();
    const __m256i mask = _mm256_setmask_pd(3);

    for (unsigned int i = 0; i < n; i++) {
        __m256d buf = zero;

        for (unsigned int s = 0; s < samples; s++) {
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
    const unsigned int n, const unsigned int samples,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (unsigned int i = 0; i < n; i++) {
        __m256d buf = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256d x = _mm256_load_pd(x_ptr);

            buf = _mm256_add_pd(x, buf);

            x_ptr += AVX2_DOUBLE_STRIDE;
        }

        _mm256_stream_pd(y_ptr, buf);

        y_ptr += AVX2_DOUBLE_STRIDE;
    }

    return SUCCESS;
}

int ag_sum_stride8_d(
    const unsigned int n, const unsigned int samples,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (unsigned int i = 0; i < n; i++) {
        __m256d buf0 = zero, buf1 = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256d x0 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;
            __m256d x1 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);
        }

        _mm256_stream_pd(y_ptr, buf0);
        y_ptr += AVX2_DOUBLE_STRIDE;
        _mm256_stream_pd(y_ptr, buf1);
        y_ptr += AVX2_DOUBLE_STRIDE;
    }

    return SUCCESS;
}

int ag_sum_stride12_d(
    const unsigned int n, const unsigned int samples,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (unsigned int i = 0; i < n; i++) {
        __m256d buf0 = zero, buf1 = zero, buf2 = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256d x0 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;
            __m256d x1 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;
            __m256d x2 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);
            buf2 = _mm256_add_pd(x2, buf2);
        }

        _mm256_stream_pd(y_ptr, buf0);
        y_ptr += AVX2_DOUBLE_STRIDE;
        _mm256_stream_pd(y_ptr, buf1);
        y_ptr += AVX2_DOUBLE_STRIDE;
        _mm256_stream_pd(y_ptr, buf2);
        y_ptr += AVX2_DOUBLE_STRIDE;
    }

    return SUCCESS;
}

int ag_sum_stride16_d(
    const unsigned int n, const unsigned int samples,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d zero = _mm256_setzero_pd();

    for (unsigned int i = 0; i < n; i++) {
        __m256d buf0 = zero, buf1 = zero, buf2 = zero, buf3 = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256d x0 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;
            __m256d x1 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;
            __m256d x2 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;
            __m256d x3 = _mm256_load_pd(x_ptr);
            x_ptr += AVX2_DOUBLE_STRIDE;

            buf0 = _mm256_add_pd(x0, buf0);
            buf1 = _mm256_add_pd(x1, buf1);
            buf2 = _mm256_add_pd(x2, buf2);
            buf3 = _mm256_add_pd(x3, buf3);
        }

        _mm256_stream_pd(y_ptr, buf0);
        y_ptr += AVX2_DOUBLE_STRIDE;
        _mm256_stream_pd(y_ptr, buf1);
        y_ptr += AVX2_DOUBLE_STRIDE;
        _mm256_stream_pd(y_ptr, buf2);
        y_ptr += AVX2_DOUBLE_STRIDE;
        _mm256_stream_pd(y_ptr, buf3);
        y_ptr += AVX2_DOUBLE_STRIDE;
    }

    return SUCCESS;
}

int ag_sum_strideleq4_d(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

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
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

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

    double* buf = (double*)_aligned_malloc(stride * sizeof(double), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(buf + c, zero);
        }

        for (unsigned int s = 0; s < samples; s++) {
            for (unsigned int c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_load_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(buf + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + c, y);
            }

            x_ptr += stride;
        }

        for (unsigned int c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            __m256d y = _mm256_load_pd(buf + c);

            _mm256_stream_pd(y_ptr + c, y);
        }

        y_ptr += stride;
    }

    _aligned_free(buf);

    return SUCCESS;
}

int ag_sum_unaligned_d(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

    if (stride <= AVX2_DOUBLE_STRIDE) {
        return ag_sum_strideleq4_d(n, samples, stride, x_ptr, y_ptr);
    }

    const unsigned int sb = stride & AVX2_DOUBLE_BATCH_MASK, sr = stride - sb;

    const __m256d zero = _mm256_setzero_pd();
    const __m256i mask = _mm256_setmask_pd(sr);

    double* buf = (double*)_aligned_malloc(((size_t)stride + AVX2_DOUBLE_STRIDE) * sizeof(double), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < stride; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(buf + c, zero);
        }

        for (unsigned int s = 0; s < samples; s++) {
            for (unsigned int c = 0; c < sb; c += AVX2_DOUBLE_STRIDE) {
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

        for (unsigned int c = 0; c < sb; c += AVX2_DOUBLE_STRIDE) {
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
    const unsigned int n, const unsigned int g, const unsigned int samples, const unsigned int stride,
    const double* __restrict x_ptr, double* __restrict y_ptr) {

    const unsigned int sg = stride * g;

#ifdef _DEBUG
    if ((sg & AVX2_DOUBLE_REMAIN_MASK) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int sb = samples / g * g, sr = samples - sb;
    const unsigned int rem = stride * sr;
    const unsigned int remb = rem & AVX2_DOUBLE_BATCH_MASK, remr = rem - remb;
    const __m256i mask = _mm256_setmask_pd(remr);

    const __m256d zero = _mm256_setzero_pd();

    double* buf = (double*)_aligned_malloc((size_t)sg * sizeof(double), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < sg; c += AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(buf + c, zero);
        }

        for (unsigned int s = 0; s < sb; s += g) {
            for (unsigned int c = 0; c < sg; c += AVX2_DOUBLE_STRIDE) {
                __m256d x = _mm256_loadu_pd(x_ptr + c);
                __m256d y = _mm256_load_pd(buf + c);

                y = _mm256_add_pd(x, y);

                _mm256_store_pd(buf + c, y);
            }
            x_ptr += sg;
        }
        if (sr > 0) {
            for (unsigned int c = 0; c < remb; c += AVX2_DOUBLE_STRIDE) {
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

    if (stride == 1u) {
#ifdef _DEBUG
        Console::WriteLine("type stride1");
#endif // _DEBUG

        ag_sum_stride1_d(n, samples, x_ptr, y_ptr);
        return;
    }

    if (stride == 2u) {
#ifdef _DEBUG
        Console::WriteLine("type stride2");
#endif // _DEBUG

        ag_sum_stride2_d(n, samples, x_ptr, y_ptr);
        return;
    }

    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        if (ag_sum_aligned_d(n, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
            throw gcnew System::OutOfMemoryException();
        }
        return;
    }

    if (stride <= MAX_AGGREGATE_BATCHING) {
        UInt32 g = Numeric::LCM(stride, AVX2_DOUBLE_STRIDE) / stride;

        if (samples >= g * 4) {
#ifdef _DEBUG
            Console::WriteLine("type batch g:" + g.ToString());
#endif // _DEBUG

            if (ag_sum_batch_d(n, g, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
                throw gcnew System::OutOfMemoryException();
            }
            return;
        }
    }

#ifdef _DEBUG
    Console::WriteLine("type unaligned");
#endif // _DEBUG

    if (ag_sum_unaligned_d(n, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
        throw gcnew System::OutOfMemoryException();
    }
    return;
}
