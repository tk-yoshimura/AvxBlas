#include "../avxblas.h"
#include "../avxblasutil.h"
#include <memory.h>

using namespace System;

void vw_alignment_fill(
    const unsigned int n, const unsigned int incx, 
    const float* __restrict v_ptr, float* __restrict y_ptr) {
    
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incx; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(v_ptr + c);

            _mm256_stream_ps(y_ptr + c, v);
        }

        y_ptr += incx;
    }
}

void vw_disorder_fill(
    const unsigned int n, const unsigned int incx, 
    const float* __restrict v_ptr, float* __restrict y_ptr) {

    const unsigned int incxb = incx & AVX2_FLOAT_BATCH_MASK, incxr = incx - incxb;

    const __m256i mask = mm256_mask(incxr);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incxb; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(v_ptr + c);

            _mm256_storeu_ps(y_ptr + c, v);
        }
        if (incxr > 0) {
            __m256 v = _mm256_maskload_ps(v_ptr + incxb, mask);

            _mm256_maskstore_ps(y_ptr + incxb, mask, v);
        }

        y_ptr += incx;
    }
}

void vw_batch_fill(
    const unsigned int n, const unsigned int g, const unsigned int incx, 
    const float* __restrict v_ptr, float* __restrict y_ptr) {
    
    const unsigned int nb = n / g * g, nr = n - nb;
    const unsigned int incxg = incx * g;

    for (unsigned int i = 0; i < nb; i += g) {
        for (unsigned int c = 0; c < incxg; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(v_ptr + c);

            _mm256_stream_ps(y_ptr + c, v);
        }

        y_ptr += incxg;
    }
    if (nr > 0) {
        const unsigned int rem = incx * nr;
        const unsigned int remb = rem & AVX2_FLOAT_BATCH_MASK, remr = rem - remb;
        const __m256i mask = mm256_mask(remr);

        for (unsigned int c = 0; c < remb; c += AVX2_FLOAT_STRIDE) {
            __m256 v = _mm256_load_ps(v_ptr + c);

            _mm256_stream_ps(y_ptr + c, v);
        }
        if (remr > 0) {
            __m256 v = _mm256_maskload_ps(v_ptr + remb, mask);

            _mm256_maskstore_ps(y_ptr + remb, mask, v);
        }
    }
}

void vw_alignment_fill(
    const unsigned int n, const unsigned int incx,
    const double* __restrict v_ptr, double* __restrict y_ptr) {

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incx; c += AVX2_DOUBLE_STRIDE) {
            __m256d v = _mm256_load_pd(v_ptr + c);

            _mm256_stream_pd(y_ptr + c, v);
        }

        y_ptr += incx;
    }
}

void vw_disorder_fill(
    const unsigned int n, const unsigned int incx,
    const double* __restrict v_ptr, double* __restrict y_ptr) {

    const unsigned int incxb = incx & AVX2_DOUBLE_BATCH_MASK, incxr = incx - incxb;

    const __m256i mask = mm256_mask(incxr * 2);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incxb; c += AVX2_DOUBLE_STRIDE) {
            __m256d v = _mm256_load_pd(v_ptr + c);

            _mm256_storeu_pd(y_ptr + c, v);
        }
        if (incxr > 0) {
            __m256d v = _mm256_maskload_pd(v_ptr + incxb, mask);

            _mm256_maskstore_pd(y_ptr + incxb, mask, v);
        }

        y_ptr += incx;
    }
}

void vw_batch_fill(
    const unsigned int n, const unsigned int g, const unsigned int incx,
    const double* __restrict v_ptr, double* __restrict y_ptr) {

    const unsigned int nb = n / g * g, nr = n - nb;
    const unsigned int incxg = incx * g;

    for (unsigned int i = 0; i < nb; i += g) {
        for (unsigned int c = 0; c < incxg; c += AVX2_DOUBLE_STRIDE) {
            __m256d v = _mm256_load_pd(v_ptr + c);

            _mm256_stream_pd(y_ptr + c, v);
        }

        y_ptr += incxg;
    }
    if (nr > 0) {
        const unsigned int rem = incx * nr;
        const unsigned int remb = rem & AVX2_DOUBLE_BATCH_MASK, remr = rem - remb;
        const __m256i mask = mm256_mask(remr * 2);

        for (unsigned int c = 0; c < remb; c += AVX2_DOUBLE_STRIDE) {
            __m256d v = _mm256_load_pd(v_ptr + c);

            _mm256_stream_pd(y_ptr + c, v);
        }
        if (remr > 0) {
            __m256d v = _mm256_maskload_pd(v_ptr + remb, mask);

            _mm256_maskstore_pd(y_ptr + remb, mask, v);
        }
    }
}

void AvxBlas::Vectorwise::Fill(UInt32 n, UInt32 incx, Array<float>^ v, Array<float>^ y) {
    if (n <= 0 || incx <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, incx);

    Util::CheckLength(n * incx, y);
    Util::CheckLength(incx, v);

    Util::CheckDuplicateArray(v, y);

    float* v_ptr = (float*)(v->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());
    
    if ((incx & AVX2_FLOAT_REMAIN_MASK) == 0u) {
        vw_alignment_fill(n, incx, v_ptr, y_ptr);
        return;
    }

    if (incx <= MAX_VECTORWISE_ALIGNMNET_INCX) {
        UInt32 ulen = lcm(incx, AVX2_FLOAT_STRIDE);
        UInt32 g = ulen / incx;

        if (n >= g * 4 && ulen <= MAX_VECTORWISE_ALIGNMNET_ULENGTH) {
            float* u_ptr = (float*)_aligned_malloc(static_cast<size_t>(ulen) * Array<float>::ElementSize, AVX2_ALIGNMENT);
            if (u_ptr == nullptr) {
                throw gcnew System::OutOfMemoryException();
            }

            alignment_vector_s(g, incx, v_ptr, u_ptr);
            vw_batch_fill(n, g, incx, u_ptr, y_ptr);

            _aligned_free(u_ptr);

            return;
        }
    }

    vw_disorder_fill(n, incx, v_ptr, y_ptr);
}

void AvxBlas::Vectorwise::Fill(UInt32 n, UInt32 incx, Array<double>^ v, Array<double>^ y) {
    if (n <= 0 || incx <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, incx);

    Util::CheckLength(n * incx, y);
    Util::CheckLength(incx, v);

    Util::CheckDuplicateArray(v, y);

    double* v_ptr = (double*)(v->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    if ((incx & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
        vw_alignment_fill(n, incx, v_ptr, y_ptr);
        return;
    }

    if (incx <= MAX_VECTORWISE_ALIGNMNET_INCX) {
        UInt32 ulen = lcm(incx, AVX2_DOUBLE_STRIDE);
        UInt32 g = ulen / incx;

        if (n >= g * 4 && ulen <= MAX_VECTORWISE_ALIGNMNET_ULENGTH) {
            double* u_ptr = (double*)_aligned_malloc(static_cast<size_t>(ulen) * Array<double>::ElementSize, AVX2_ALIGNMENT);
            if (u_ptr == nullptr) {
                throw gcnew System::OutOfMemoryException();
            }

            alignment_vector_d(g, incx, v_ptr, u_ptr);
            vw_batch_fill(n, g, incx, u_ptr, y_ptr);

            _aligned_free(u_ptr);

            return;
        }
    }

    vw_disorder_fill(n, incx, v_ptr, y_ptr);
}
