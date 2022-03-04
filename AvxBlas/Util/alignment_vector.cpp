#include "../AvxBlas.h"
#include "../AvxBlasUtil.h"

#include <immintrin.h>

void AvxBlas::alignment_vector_s(
    const unsigned int n, const unsigned int incx, 
    const float* __restrict x_ptr, float* __restrict y_ptr) {
    
    const unsigned int incxb = incx & AVX2_FLOAT_BATCH_MASK, incxr = incx - incxb;
    
    mm256_mask(const __m256i mask, incxr);
    
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incxb; c += AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_load_ps(x_ptr + c);

            _mm256_storeu_ps(y_ptr + c, x);
        }
        if (incxr > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr + incxb, mask);

            _mm256_maskstore_ps(y_ptr + incxb, mask, x);
        }

        y_ptr += incx;
    }
}

void AvxBlas::alignment_vector_d(
    const unsigned int n, const unsigned int incx, 
    const double* __restrict x_ptr, double* __restrict y_ptr) {
    
    const unsigned int incxb = incx & AVX2_DOUBLE_BATCH_MASK, incxr = incx - incxb;

    mm256_mask(const __m256i mask, incxr * 2);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int c = 0; c < incxb; c += AVX2_DOUBLE_STRIDE) {
            __m256d x = _mm256_load_pd(x_ptr + c);

            _mm256_storeu_pd(y_ptr + c, x);
        }
        if (incxr > 0) {
            __m256d x = _mm256_maskload_pd(x_ptr + incxb, mask);

            _mm256_maskstore_pd(y_ptr + incxb, mask, x);
        }

        y_ptr += incx;
    }
}