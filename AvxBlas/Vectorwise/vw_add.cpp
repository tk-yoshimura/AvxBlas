#include "../AvxBlas.h"
#include "../AvxBlasUtil.h"
#include <memory.h>

using namespace System;

void vw_alignment_add(unsigned int n, unsigned int incx, const float* __restrict x_ptr, const float* __restrict v_ptr, float* __restrict y_ptr) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incx; c += 8) {
            __m256 x = _mm256_load_ps(x_ptr + c);
            __m256 v = _mm256_load_ps(v_ptr + c);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_stream_ps(y_ptr + c, y);
        }

        x_ptr += incx;
        y_ptr += incx;
    }
}

void vw_disorder_add(unsigned int n, unsigned int incx, const float* __restrict x_ptr, const float* __restrict v_ptr, float* __restrict y_ptr) {
    const unsigned int incxb = incx & ~7u, incxr = incx - incxb;

    const __m256i mask = AvxBlas::masktable_m256(incxr);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incxb; c += 8) {
            __m256 x = _mm256_loadu_ps(x_ptr + c);
            __m256 v = _mm256_load_ps(v_ptr + c);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_storeu_ps(y_ptr + c, y);
        }
        if (incxr > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr + incxb, mask);
            __m256 v = _mm256_maskload_ps(v_ptr + incxb, mask);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_maskstore_ps(y_ptr + incxb, mask, y);
        }

        x_ptr += incx;
        y_ptr += incx;
    }
}

void vw_batch_add(unsigned int n, unsigned int g, unsigned int incx, const float* __restrict x_ptr, const float* __restrict v_ptr, float* __restrict y_ptr) {
    const unsigned int nb = n / g * g, nr = n - nb;
    const unsigned int incxg = incx * g;

    for (unsigned int i = 0; i < nb; i += g) {
        for (unsigned int c = 0; c < incxg; c += 8) {
            __m256 x = _mm256_load_ps(x_ptr + c);
            __m256 v = _mm256_load_ps(v_ptr + c);

            __m256 y = _mm256_add_ps(x, v);

            _mm256_stream_ps(y_ptr + c, y);
        }

        x_ptr += incxg;
        y_ptr += incxg;
    }
    if (nr > 0) {
        const unsigned int rem = incx * nr;
        const unsigned int remb = rem & ~7u, remr = rem - remb;
        const __m256i mask = AvxBlas::masktable_m256(remr);

        for (unsigned int c = 0; c < remb; c += 8) {
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

void AvxBlas::Vectorwise::Add(UInt32 n, UInt32 incx, Array<float>^ x, Array<float>^ v, Array<float>^ y) {
    if (n <= 0 || incx <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, incx);

    Util::CheckLength(n * incx, x, y);
    Util::CheckLength(incx, v);

    Util::CheckDuplicateArray(v, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* v_ptr = (float*)(v->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());
    
    if ((incx & 7u) == 0u) {
        vw_alignment_add(n, incx, x_ptr, v_ptr, y_ptr);
        return;
    }

    if (incx <= MAX_VECTORWISE_ALIGNMNET_INCX) {
        UInt32 ulen = lcm(incx, AVX2_ALIGNMENT / Array<float>::ElementSize);
        UInt32 g = ulen / incx;

        if (n >= g * 4 && ulen <= MAX_VECTORWISE_ALIGNMNET_ULENGTH) {
            float* u_ptr = (float*)_aligned_malloc(static_cast<size_t>(ulen) * Array<float>::ElementSize, AVX2_ALIGNMENT);
            if (u_ptr == nullptr) {
                throw gcnew System::OutOfMemoryException();
            }

            alignment_vector_s(g, incx, v_ptr, u_ptr);
            vw_batch_add(n, g, incx, x_ptr, u_ptr, y_ptr);

            _aligned_free(u_ptr);

            return;
        }
    }

    vw_disorder_add(n, incx, x_ptr, v_ptr, y_ptr);
}

//void AvxBlas::Vectorwise::Add(UInt32 n, UInt32 incx, Array<double>^ x, Array<double>^ v, Array<double>^ y) {
//    Util::CheckLength(n * incx, x, y);
//    Util::CheckLength(incx, v);
//
//    double* x_ptr = (double*)(x->Ptr.ToPointer());
//    double* v_ptr = (double*)(v->Ptr.ToPointer());
//    double* y_ptr = (double*)(y->Ptr.ToPointer());
//
//    vw_alignment_add(n, x1_ptr, x2_ptr, y_ptr);
//}
