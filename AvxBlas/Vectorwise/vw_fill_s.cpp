#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include <memory.h>

using namespace System;

#pragma unmanaged

int vw_fill_aligned_s(
    const unsigned int n, const unsigned int stride,
    const float* __restrict v_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < stride; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(v_ptr + c);

            _mm256_stream_ps(y_ptr + c, v);
        }

        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_fill_unaligned_s(
    const unsigned int n, const unsigned int stride,
    const float* __restrict v_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int sb = stride & AVX2_FLOAT_BATCH_MASK, sr = stride - sb;

    const __m256i mask = _mm256_setmask_ps(sr);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < sb; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(v_ptr + c);

            _mm256_storeu_ps(y_ptr + c, v);
        }
        if (sr > 0) {
            __m256 v = _mm256_maskload_ps(v_ptr + sb, mask);

            _mm256_maskstore_ps(y_ptr + sb, mask, v);
        }

        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_fill_batch_s(
    const unsigned int n, const unsigned int g, const unsigned int stride,
    const float* __restrict v_ptr, float* __restrict y_ptr) {

    const unsigned int nb = n / g * g, nr = n - nb;
    const unsigned int sg = stride * g;

#ifdef _DEBUG
    if ((sg & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* u_ptr = (float*)_aligned_malloc(sg * sizeof(float), AVX2_ALIGNMENT);
    if (u_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    repeat_vector_s(g, stride, v_ptr, u_ptr);

    for (unsigned int i = 0; i < nb; i += g) {
        for (unsigned int c = 0; c < sg; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(u_ptr + c);

            _mm256_stream_ps(y_ptr + c, v);
        }

        y_ptr += sg;
    }
    if (nr > 0) {
        const unsigned int rem = stride * nr;
        const unsigned int remb = rem & AVX2_FLOAT_BATCH_MASK, remr = rem - remb;
        const __m256i mask = _mm256_setmask_ps(remr);

        for (unsigned int c = 0; c < remb; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(u_ptr + c);

            _mm256_stream_ps(y_ptr + c, v);
        }
        if (remr > 0) {
            __m256 v = _mm256_maskload_ps(u_ptr + remb, mask);

            _mm256_maskstore_ps(y_ptr + remb, mask, v);
        }
    }

    _aligned_free(u_ptr);

    return SUCCESS;
}

#pragma managed

void AvxBlas::Vectorwise::Fill(UInt32 n, UInt32 stride, Array<float>^ v, Array<float>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, stride);

    Util::CheckLength(n * stride, y);
    Util::CheckLength(stride, v);

    Util::CheckDuplicateArray(v, y);

    const float* v_ptr = (const float*)(v->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        vw_fill_aligned_s(n, stride, v_ptr, y_ptr);
        return;
    }

    if (stride <= MAX_VECTORWISE_ALIGNMNET_INCX) {
        UInt32 g = Numeric::LCM(stride, AVX2_FLOAT_STRIDE) / stride;

        if (n >= g * 4) {
#ifdef _DEBUG
            Console::WriteLine("type batch g:" + g.ToString());
#endif // _DEBUG

            if (vw_fill_batch_s(n, g, stride, v_ptr, y_ptr) == FAILURE_BADALLOC) {
                throw gcnew System::OutOfMemoryException();
            }
            return;
        }
    }

#ifdef _DEBUG
    Console::WriteLine("type unaligned");
#endif // _DEBUG

    vw_fill_unaligned_s(n, stride, v_ptr, y_ptr);
}