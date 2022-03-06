#include "../avxblas.h"
#include "../avxblasutil.h"
#include <memory.h>
#include <exception>
#include <stdexcept>

using namespace System;

void ag_lessstride_sum_s(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride > AVX2_FLOAT_STRIDE) {
        throw std::exception();
    }
#endif // _DEBUG

    if (stride == AVX2_FLOAT_STRIDE) {
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
    }
    else if (stride > AVX2_FLOAT_STRIDE / 2) {
        const __m256 zero = _mm256_setzero_ps();
        const __m256i mask = mm256_mask(stride);

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
    }
    else if (stride == AVX2_FLOAT_STRIDE / 2) {
        const __m256 zero = _mm256_setzero_ps();
        const unsigned int sb = samples / 2 * 2, sr = samples - sb;
        const __m256i mask4 = mm256_mask(4);

        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += 2) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
            if (sr > 0) {
                __m256 x = _mm256_maskload_ps(x_ptr, mask4);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE / 2;
            }

            __m128 y = _mm_add_ps(_mm256_extractf128_ps(buf, 0), _mm256_extractf128_ps(buf, 1));

            _mm_stream_ps(y_ptr, y);

            y_ptr += AVX2_FLOAT_STRIDE / 2;
        }
    }
    else if (stride == 3) {
        const __m256 zero = _mm256_setzero_ps();
        const unsigned int sb = samples / 2 * 2, sr = samples - sb;
        const __m128i mask3 = mm128_mask(3);
        const __m256i mask6 = mm256_mask(6);
        const __m256i perm = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

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

            buf = _mm256_permutevar8x32_ps(buf, perm);

            __m128 y = _mm_add_ps(_mm256_extractf128_ps(buf, 0), _mm256_extractf128_ps(buf, 1));

            _mm_maskstore_ps(y_ptr, mask3, y);

            y_ptr += 3;
        }
    }
    else if (stride == 2) {
        const __m256 zero = _mm256_setzero_ps();
        const unsigned int sb = samples / 4 * 4, sr = samples - sb;
        const __m128i mask2 = mm128_mask(2);
        const __m256i mask = mm256_mask(sr * 2);

        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += 4) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
            if (sr > 0) {
                __m256 x = _mm256_maskload_ps(x_ptr, mask);

                buf = _mm256_add_ps(x, buf);

                x_ptr += sr * 2;
            }

            __m128 y = _mm_add_ps(_mm256_extractf128_ps(buf, 0), _mm256_extractf128_ps(buf, 1));
            y = _mm_add_ps(y, _mm_permute_ps(y, _MM_PERM_BADC));

            _mm_maskstore_ps(y_ptr, mask2, y);

            y_ptr += 2;
        }
    }
    else if (stride == 1) {
        const __m256 zero = _mm256_setzero_ps();
        const unsigned int sb = samples & AVX2_FLOAT_BATCH_MASK, sr = samples - sb;
        const __m128i mask1 = mm128_mask(1);
        const __m256i mask = mm256_mask(sr);

        for (unsigned int i = 0; i < n; i++) {
            __m256 buf = zero;

            for (unsigned int s = 0; s < sb; s += AVX2_FLOAT_STRIDE) {
                __m256 x = _mm256_loadu_ps(x_ptr);

                buf = _mm256_add_ps(x, buf);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
            if (sr > 0) {
                __m256 x = _mm256_maskload_ps(x_ptr, mask);

                buf = _mm256_add_ps(x, buf);

                x_ptr += sr;
            }

            __m128 y = _mm_add_ps(_mm256_extractf128_ps(buf, 0), _mm256_extractf128_ps(buf, 1));
            y = _mm_add_ps(y, _mm_permute_ps(y, _MM_PERM_CDAB));
            y = _mm_add_ps(y, _mm_permute_ps(y, _MM_PERM_BADC));

            _mm_maskstore_ps(y_ptr, mask1, y);

            y_ptr += 1;
        }
    }
#ifdef _DEBUG
    else {
        throw std::exception();
    }
#endif // _DEBUG
}

void ag_alignment_sum_s(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    if (stride == AVX2_FLOAT_STRIDE) {
        ag_lessstride_sum_s(n, samples, stride, x_ptr, y_ptr);
        return;
    }

    const __m256 zero = _mm256_setzero_ps();

    float* buf = (float*)_aligned_malloc(stride * sizeof(float), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        throw std::bad_alloc();
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
}

void ag_disorder_sum_s(
    const unsigned int n, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    if (stride <= AVX2_FLOAT_STRIDE) {
        ag_lessstride_sum_s(n, samples, stride, x_ptr, y_ptr);
        return;
    }

    const unsigned int sb = stride & AVX2_FLOAT_BATCH_MASK, sr = stride - sb;

    const __m256 zero = _mm256_setzero_ps();
    const __m256i mask = mm256_mask(sr);

    float* buf = (float*)_aligned_malloc(((size_t)stride + AVX2_FLOAT_STRIDE) * sizeof(float), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        throw std::bad_alloc();
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
}

void ag_batch_sum_s(
    const unsigned int n, const unsigned int g, const unsigned int samples, const unsigned int stride,
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    const unsigned int sg = stride * g;

#ifdef _DEBUG
    if((sg & AVX2_FLOAT_REMAIN_MASK) != 0){
        throw std::exception();
    }
#endif // _DEBUG

    const unsigned int sb = samples / g * g, sr = samples - sb;
    const unsigned int rem = stride * sr;
    const unsigned int remb = rem & AVX2_FLOAT_BATCH_MASK, remr = rem - remb;
    const __m256i mask = mm256_mask(remr);

    const __m256 zero = _mm256_setzero_ps();

    float* buf = (float*)_aligned_malloc((size_t)sg * sizeof(float), AVX2_ALIGNMENT);
    if (buf == nullptr) {
        throw std::bad_alloc();
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
}

void AvxBlas::Aggregate::Sum(UInt32 n, UInt32 samples, UInt32 stride, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || samples <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, samples, stride);

    Util::CheckLength(n * samples * stride, x);
    Util::CheckLength(n * stride, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    try {
        if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
            Console::WriteLine("type alignment");
#endif // _DEBUG

            ag_alignment_sum_s(n, samples, stride, x_ptr, y_ptr);
            return;
        }

        if (stride <= MAX_AGGREGATE_BATCHING) {
            UInt32 g = lcm(stride, AVX2_FLOAT_STRIDE) / stride;

            if (samples >= g * 4) {
#ifdef _DEBUG
                Console::WriteLine("type batch g:" + g.ToString());
#endif // _DEBUG

                ag_batch_sum_s(n, g, samples, stride, x_ptr, y_ptr);
                return;
            }
        }

#ifdef _DEBUG
        Console::WriteLine("type disorder");
#endif // _DEBUG

        ag_disorder_sum_s(n, samples, stride, x_ptr, y_ptr);
    }
    catch (std::bad_alloc) {
        throw gcnew System::OutOfMemoryException();
    }
}
