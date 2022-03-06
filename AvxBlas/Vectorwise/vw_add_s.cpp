#include "../avxblas.h"
#include "../avxblasutil.h"
#include <memory.h>

using namespace System;

void vw_alignment_add_s(
    const unsigned int n, const unsigned int stride, 
    const float* __restrict x_ptr, const float* __restrict v_ptr, float* __restrict y_ptr) {
    
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_load_ps(x_ptr + c);
            __m256 v = _mm256_load_ps(v_ptr + c);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_stream_ps(y_ptr + c, y);
        }

        x_ptr += stride;
        y_ptr += stride;
    }
}

void vw_disorder_add_s(
    const unsigned int n, const unsigned int stride, 
    const float* __restrict x_ptr, const float* __restrict v_ptr, float* __restrict y_ptr) {

    const unsigned int sb = stride & AVX2_FLOAT_BATCH_MASK, sr = stride - sb;

    const __m256i mask = mm256_mask(sr);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < sb; c += AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_loadu_ps(x_ptr + c);
            __m256 v = _mm256_load_ps(v_ptr + c);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_storeu_ps(y_ptr + c, y);
        }
        if (sr > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr + sb, mask);
            __m256 v = _mm256_maskload_ps(v_ptr + sb, mask);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_maskstore_ps(y_ptr + sb, mask, y);
        }

        x_ptr += stride;
        y_ptr += stride;
    }
}

void vw_batch_add_s(
    const unsigned int n, const unsigned int g, const unsigned int stride, 
    const float* __restrict x_ptr, const float* __restrict v_ptr, float* __restrict y_ptr) {
    
    const unsigned int nb = n / g * g, nr = n - nb;
    const unsigned int sg = stride * g;

    for (unsigned int i = 0; i < nb; i += g) {
        for (unsigned int c = 0; c < sg; c += AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_load_ps(x_ptr + c);
            __m256 v = _mm256_load_ps(v_ptr + c);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_stream_ps(y_ptr + c, y);
        }

        x_ptr += sg;
        y_ptr += sg;
    }
    if (nr > 0) {
        const unsigned int rem = stride * nr;
        const unsigned int remb = rem & AVX2_FLOAT_BATCH_MASK, remr = rem - remb;
        const __m256i mask = mm256_mask(remr);

        for (unsigned int c = 0; c < remb; c += AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_load_ps(x_ptr + c);
            __m256 v = _mm256_load_ps(v_ptr + c);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_stream_ps(y_ptr + c, y);
        }
        if (remr > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr + remb, mask);
            __m256 v = _mm256_maskload_ps(v_ptr + remb, mask);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_maskstore_ps(y_ptr + remb, mask, y);
        }
    }
}

void AvxBlas::Vectorwise::Add(UInt32 n, UInt32 stride, Array<float>^ x, Array<float>^ v, Array<float>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, stride);

    Util::CheckLength(n * stride, x, y);
    Util::CheckLength(stride, v);

    Util::CheckDuplicateArray(v, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* v_ptr = (float*)(v->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());
    
    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
        vw_alignment_add_s(n, stride, x_ptr, v_ptr, y_ptr);
        return;
    }

    if (stride <= MAX_VECTORWISE_ALIGNMNET_INCX) {
        UInt32 ulen = lcm(stride, AVX2_FLOAT_STRIDE);
        UInt32 g = ulen / stride;

        if (n >= g * 4 && ulen <= MAX_VECTORWISE_ALIGNMNET_ULENGTH) {
            float* u_ptr = (float*)_aligned_malloc(static_cast<size_t>(ulen) * Array<float>::ElementSize, AVX2_ALIGNMENT);
            if (u_ptr == nullptr) {
                throw gcnew System::OutOfMemoryException();
            }

            alignment_vector_s(g, stride, v_ptr, u_ptr);
            vw_batch_add_s(n, g, stride, x_ptr, u_ptr, y_ptr);

            _aligned_free(u_ptr);

            return;
        }
    }

    vw_disorder_add_s(n, stride, x_ptr, v_ptr, y_ptr);
}