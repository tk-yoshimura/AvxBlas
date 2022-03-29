#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"

#pragma unmanaged

void zeroset_s(const uint n, outfloats y_ptr) {
    uint r = n;

    while (r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = 0.0f;
        y_ptr++;
        r--;
    }

    const __m256 fillz = _mm256_setzero_ps();

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_stream_x4_ps(y_ptr, fillz, fillz, fillz, fillz);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_stream_x2_ps(y_ptr, fillz, fillz);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_stream_x1_ps(y_ptr, fillz);

        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        _mm256_maskstore_ps(y_ptr, mask, fillz);
    }
}

void zeroset_d(const uint n, outdoubles y_ptr) {
    uint r = n;

    while (r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = 0.0;
        y_ptr++;
        r--;
    }

    const __m256d fillz = _mm256_setzero_pd();

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_stream_x4_pd(y_ptr, fillz, fillz, fillz, fillz);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_stream_x2_pd(y_ptr, fillz, fillz);

        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_stream_x1_pd(y_ptr, fillz);

        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        _mm256_maskstore_pd(y_ptr, mask, fillz);
    }
}