#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

#pragma unmanaged

void repeat_vector_s(
    const uint n, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    const uint sb = stride & AVX2_FLOAT_BATCH_MASK, sr = stride - sb;

    const __m256i mask = _mm256_setmask_ps(sr);

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < sb; c += AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_load_ps(x_ptr + c);

            _mm256_storeu_ps(y_ptr + c, x);
        }
        if (sr > 0) {
            __m256 x = _mm256_maskload_ps(x_ptr + sb, mask);

            _mm256_maskstore_ps(y_ptr + sb, mask, x);
        }

        y_ptr += stride;
    }
}

void repeat_vector_d(
    const uint n, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    const uint sb = stride & AVX2_DOUBLE_BATCH_MASK, sr = stride - sb;

    const __m256i mask = _mm256_setmask_pd(sr);

    for (uint i = 0; i < n; i++) {
        for (uint c = 0; c < sb; c += AVX2_DOUBLE_STRIDE) {
            __m256d x = _mm256_load_pd(x_ptr + c);

            _mm256_storeu_pd(y_ptr + c, x);
        }
        if (sr > 0) {
            __m256d x = _mm256_maskload_pd(x_ptr + sb, mask);

            _mm256_maskstore_pd(y_ptr + sb, mask, x);
        }

        y_ptr += stride;
    }
}