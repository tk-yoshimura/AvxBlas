#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_sum.cpp"
#include <memory.h>

using namespace System;

#pragma unmanaged

int ag_stride1_sum_s(
    const unsigned int n, const unsigned int samples,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const unsigned int sb = samples & AVX2_FLOAT_BATCH_MASK, sr = samples - sb;
    const __m256i mask = _mm256_set_mask(sr);

    if (sr > 0) {
        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
            {
                __m256 x = _mm256_maskload_ps(x_ptr, mask);

                buf = _mm256_add_ps(x, buf);

                x_ptr += sr;
            }

            float y = _mm256_sum8to1_ps(buf);

            *y_ptr = y;

            y_ptr += 1;
        }
    }
    else {
        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_load_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }

            float y = _mm256_sum8to1_ps(buf);

            *y_ptr = y;

            y_ptr += 1;
        }
    }

    return SUCCESS;
}

int ag_stride2_sum_s(
    const unsigned int n, const unsigned int samples,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const unsigned int sb = samples / 4 * 4, sr = samples - sb;
    const __m128i mask2 = _mm_set_mask(2);
    const __m256i mask = _mm256_set_mask(sr * 2);

    if (sr > 0) {
        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += 4) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
            {
                __m256 x = _mm256_maskload_ps(x_ptr, mask);

                buf = _mm256_add_ps(x, buf);

                x_ptr += sr * 2;
            }

            __m128 y = _mm256_sum8to2_ps(buf);

            _mm_maskstore_ps(y_ptr, mask2, y);

            y_ptr += 2;
        }
    }
    else {
        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += 4) {
                __m256 x = _mm256_load_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }

            __m128 y = _mm256_sum8to2_ps(buf);

            _mm_maskstore_ps(y_ptr, mask2, y);

            y_ptr += 2;
        }
    }

    return SUCCESS;
}

int ag_stride3_sum_s(
    const unsigned int n, const unsigned int samples,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    const __m256 zero = _mm256_setzero_ps();
    const unsigned int sb = samples / 2 * 2, sr = samples - sb;
    const __m128i mask3 = _mm_set_mask(3);
    const __m256i mask6 = _mm256_set_mask(6);

    for (unsigned int i = 0; i < n; i++) {
        __m256 buf = zero;

        for (unsigned int s = 0; s < sb; s += 2) {
            __m256 x = _mm256_maskload_ps(x_ptr, mask6);

            buf = _mm256_add_ps(x, buf);

            x_ptr += 6;
        }
        if (sr > 0) {
            __m256 x = _mm256_castps128_ps256(_mm_maskload_ps(x_ptr, mask3));

            buf = _mm256_add_ps(x, buf);

            x_ptr += 3;
        }

        __m128 y = _mm256_sum6to3_ps(buf);

        _mm_maskstore_ps(y_ptr, mask3, y);

        y_ptr += 3;
    }

    return SUCCESS;
}

int ag_stride4_sum_s(
    const unsigned int n, const unsigned int samples,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();
    const unsigned int sb = samples / 2 * 2, sr = samples - sb;
    const __m256i mask4 = _mm256_set_mask(4);

    if (sr > 0) {
        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += 2) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
            {
                __m256 x = _mm256_maskload_ps(x_ptr, mask4);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE / 2;
            }

            __m128 y = _mm256_sum8to4_ps(buf);

            _mm_stream_ps(y_ptr, y);

            y_ptr += AVX2_FLOAT_STRIDE / 2;
        }
    }
    else {
        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += 2) {
                __m256 x = _mm256_load_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }

            __m128 y = _mm256_sum8to4_ps(buf);

            _mm_stream_ps(y_ptr, y);

            y_ptr += AVX2_FLOAT_STRIDE / 2;
        }
    }

    return SUCCESS;
}

int ag_stride5to7_sum_s(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE / 2 || stride >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = _mm256_set_mask(stride);

    for (unsigned int i = 0; i < n; i++) {
        __m256 buf = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256 x = _mm256_maskload_ps(x_ptr, mask);

            buf = _mm256_add_ps(x, buf);

            x_ptr += stride;
        }

        _mm256_maskstore_ps(y_ptr, mask, buf);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_stride8_sum_s(
    const unsigned int n, const unsigned int samples, 
    const float* __restrict x_ptr, float* __restrict y_ptr){

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (unsigned int i = 0; i < n; i++) {
        __m256 buf = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256 x = _mm256_load_ps(x_ptr);

            buf = _mm256_add_ps(x, buf);

            x_ptr += AVX2_FLOAT_STRIDE;
        }

        _mm256_stream_ps(y_ptr, buf);

        y_ptr += AVX2_FLOAT_STRIDE;
    }

    return SUCCESS;
}

int ag_stride16_sum_s(
    const unsigned int n, const unsigned int samples,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (unsigned int i = 0; i < n; i++) {
        __m256 buf0 = zero, buf1 = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256 x0 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;
            __m256 x1 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;

            buf0 = _mm256_add_ps(x0, buf0);
            buf1 = _mm256_add_ps(x1, buf1);
        }

        _mm256_stream_ps(y_ptr, buf0);
        y_ptr += AVX2_FLOAT_STRIDE;
        _mm256_stream_ps(y_ptr, buf1);
        y_ptr += AVX2_FLOAT_STRIDE;
    }

    return SUCCESS;
}

int ag_stride24_sum_s(
    const unsigned int n, const unsigned int samples,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (unsigned int i = 0; i < n; i++) {
        __m256 buf0 = zero, buf1 = zero, buf2 = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256 x0 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;
            __m256 x1 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;
            __m256 x2 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;

            buf0 = _mm256_add_ps(x0, buf0);
            buf1 = _mm256_add_ps(x1, buf1);
            buf2 = _mm256_add_ps(x2, buf2);
        }

        _mm256_stream_ps(y_ptr, buf0);
        y_ptr += AVX2_FLOAT_STRIDE;
        _mm256_stream_ps(y_ptr, buf1);
        y_ptr += AVX2_FLOAT_STRIDE;
        _mm256_stream_ps(y_ptr, buf2);
        y_ptr += AVX2_FLOAT_STRIDE;
    }

    return SUCCESS;
}

int ag_stride32_sum_s(
    const unsigned int n, const unsigned int samples,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (unsigned int i = 0; i < n; i++) {
        __m256 buf0 = zero, buf1 = zero, buf2 = zero, buf3 = zero;

        for (unsigned int s = 0; s < samples; s++) {
            __m256 x0 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;
            __m256 x1 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;
            __m256 x2 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;
            __m256 x3 = _mm256_load_ps(x_ptr);
            x_ptr += AVX2_FLOAT_STRIDE;

            buf0 = _mm256_add_ps(x0, buf0);
            buf1 = _mm256_add_ps(x1, buf1);
            buf2 = _mm256_add_ps(x2, buf2);
            buf3 = _mm256_add_ps(x3, buf3);
        }

        _mm256_stream_ps(y_ptr, buf0);
        y_ptr += AVX2_FLOAT_STRIDE;
        _mm256_stream_ps(y_ptr, buf1);
        y_ptr += AVX2_FLOAT_STRIDE;
        _mm256_stream_ps(y_ptr, buf2);
        y_ptr += AVX2_FLOAT_STRIDE;
        _mm256_stream_ps(y_ptr, buf3);
        y_ptr += AVX2_FLOAT_STRIDE;
    }

    return SUCCESS;
}

int ag_lessstride_sum_s(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    if (stride == 1) {
        return ag_stride1_sum_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 2) {
        return ag_stride2_sum_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == 3) {
        return ag_stride3_sum_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride == AVX2_FLOAT_STRIDE / 2) {
        return ag_stride4_sum_s(n, samples, x_ptr, y_ptr);
    }
    else if (stride > AVX2_FLOAT_STRIDE / 2) {
        return ag_stride5to7_sum_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == AVX2_FLOAT_STRIDE) {
        return ag_stride8_sum_s(n, samples, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int ag_alignment_sum_s(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    if (stride == AVX2_FLOAT_STRIDE) {
        return ag_stride8_sum_s(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 2) {
        return ag_stride16_sum_s(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 3) {
        return ag_stride24_sum_s(n, samples, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 4) {
        return ag_stride32_sum_s(n, samples, x_ptr, y_ptr);
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    float* buf = (float*)_aligned_malloc(stride * sizeof(float), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            _mm256_store_ps(buf + c, zero);
        }

        for (unsigned int s = 0; s < samples; s++) {
            for (unsigned int c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_load_ps(x_ptr + c);
                __m256 y = _mm256_load_ps(buf + c);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(buf + c, y);
            }

            x_ptr += stride;
        }

        for (unsigned int c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            __m256 y = _mm256_load_ps(buf + c);

            _mm256_stream_ps(y_ptr + c, y);
        }

        y_ptr += stride;
    }

    _aligned_free(buf);

    return SUCCESS;
}

int ag_disorder_sum_s(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    if (stride <= AVX2_FLOAT_STRIDE) {
        return ag_lessstride_sum_s(n, samples, stride, x_ptr, y_ptr);
    }

    const unsigned int sb = stride & AVX2_FLOAT_BATCH_MASK, sr = stride - sb;

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = _mm256_set_mask(sr);

    float* buf = (float*)_aligned_malloc(((size_t)stride + AVX2_FLOAT_STRIDE) * sizeof(float), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            _mm256_store_ps(buf + c, zero);
        }

        for (unsigned int s = 0; s < samples; s++) {
            for (unsigned int c = 0; c < sb; c += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr + c);
                __m256 y = _mm256_load_ps(buf + c);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(buf + c, y);
            }
            if (sr > 0) {
                __m256 x = _mm256_maskload_ps(x_ptr + sb, mask);
                __m256 y = _mm256_load_ps(buf + sb);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(buf + sb, y);
            }

            x_ptr += stride;
        }

        for (unsigned int c = 0; c < sb; c += AVX2_FLOAT_STRIDE) {
            __m256 y = _mm256_load_ps(buf + c);

            _mm256_storeu_ps(y_ptr + c, y);
        }
        if (sr > 0) {
            __m256 y = _mm256_load_ps(buf + sb);

            _mm256_maskstore_ps(y_ptr + sb, mask, y);
        }

        y_ptr += stride;
    }

    _aligned_free(buf);

    return SUCCESS;
}

int ag_batch_sum_s(
    const unsigned int n, const unsigned int g, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    const unsigned int sg = stride * g;

#ifdef _DEBUG
    if((sg & AVX2_FLOAT_REMAIN_MASK) != 0){
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int sb = samples / g * g, sr = samples - sb;
    const unsigned int rem = stride * sr;
    const unsigned int remb = rem & AVX2_FLOAT_BATCH_MASK, remr = rem - remb;
    const __m256i mask = _mm256_set_mask(remr);

    const __m256 zero = _mm256_setzero_ps();

    float* buf = (float*)_aligned_malloc((size_t)sg * sizeof(float), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < sg; c += AVX2_FLOAT_STRIDE) {
            _mm256_store_ps(buf + c, zero);
        }

        for (unsigned int s = 0; s < sb; s += g) {
            for (unsigned int c = 0; c < sg; c += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr + c);
                __m256 y = _mm256_load_ps(buf + c);

                y = _mm256_add_ps(x, y);

                _mm256_store_ps(buf + c, y);
            }
            x_ptr += sg;
        }
        if (sr > 0) {
            for (unsigned int c = 0; c < remb; c += AVX2_FLOAT_STRIDE) {
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

        ag_disorder_sum_s(1, g, stride, buf, y_ptr);

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

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    if (stride == 1u) {
#ifdef _DEBUG
        Console::WriteLine("type stride1");
#endif // _DEBUG

        ag_stride1_sum_s(n, samples, x_ptr, y_ptr);
        return;
    }

    if (stride == 2u) {
#ifdef _DEBUG
        Console::WriteLine("type stride2");
#endif // _DEBUG

        ag_stride2_sum_s(n, samples, x_ptr, y_ptr);
        return;
    }

    if (stride == 4u) {
#ifdef _DEBUG
        Console::WriteLine("type stride4");
#endif // _DEBUG

        ag_stride4_sum_s(n, samples, x_ptr, y_ptr);
        return;
    }

    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type alignment");
#endif // _DEBUG

        if (ag_alignment_sum_s(n, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
            throw gcnew System::OutOfMemoryException();
        }
        return;
    }

    if (stride <= MAX_AGGREGATE_BATCHING) {
        UInt32 g = Util::LCM(stride, AVX2_FLOAT_STRIDE) / stride;

        if (samples >= g * 4) {
#ifdef _DEBUG
            Console::WriteLine("type batch g:" + g.ToString());
#endif // _DEBUG

            if (ag_batch_sum_s(n, g, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
                throw gcnew System::OutOfMemoryException();
            }
            return;
        }
    }

#ifdef _DEBUG
    Console::WriteLine("type disorder");
#endif // _DEBUG

    if (ag_disorder_sum_s(n, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
        throw gcnew System::OutOfMemoryException();
    }
    return;
}
