#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"

#pragma unmanaged

void copy_s(const unsigned int n, infloats x_ptr, outfloats y_ptr) {
    unsigned int r = n;

    while (r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = *x_ptr;
        x_ptr++;
        y_ptr++;
        r--;
    }

    __m256 x0, x1, x2, x3;
    
    if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
            _mm256_stream_x4_ps(y_ptr, x0, x1, x2, x3);

            x_ptr += AVX2_FLOAT_STRIDE * 4;
            y_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(x_ptr, x0, x1);
            _mm256_stream_x2_ps(y_ptr, x0, x1);

            x_ptr += AVX2_FLOAT_STRIDE * 2;
            y_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(x_ptr, x0);
            _mm256_stream_x1_ps(y_ptr, x0);

            x_ptr += AVX2_FLOAT_STRIDE;
            y_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
    }
    else {
        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
            _mm256_stream_x4_ps(y_ptr, x0, x1, x2, x3);

            x_ptr += AVX2_FLOAT_STRIDE * 4;
            y_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(x_ptr, x0, x1);
            _mm256_stream_x2_ps(y_ptr, x0, x1);

            x_ptr += AVX2_FLOAT_STRIDE * 2;
            y_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(x_ptr, x0);
            _mm256_stream_x1_ps(y_ptr, x0);

            x_ptr += AVX2_FLOAT_STRIDE;
            y_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
    }

    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x0 = _mm256_maskload_ps(x_ptr, mask);
        _mm256_maskstore_ps(y_ptr, mask, x0);
    }
}

void copy_d(const unsigned int n, indoubles x_ptr, outdoubles y_ptr) {
    unsigned int r = n;

    while (r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = *x_ptr;
        x_ptr++;
        y_ptr++;
        r--;
    }

    __m256d x0, x1, x2, x3;

    if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);
            _mm256_stream_x4_pd(y_ptr, x0, x1, x2, x3);

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
            y_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(x_ptr, x0, x1);
            _mm256_stream_x2_pd(y_ptr, x0, x1);

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
            y_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(x_ptr, x0);
            _mm256_stream_x1_pd(y_ptr, x0);

            x_ptr += AVX2_DOUBLE_STRIDE;
            y_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
    }
    else {
        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);
            _mm256_stream_x4_pd(y_ptr, x0, x1, x2, x3);

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
            y_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(x_ptr, x0, x1);
            _mm256_stream_x2_pd(y_ptr, x0, x1);

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
            y_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(x_ptr, x0);
            _mm256_stream_x1_pd(y_ptr, x0);

            x_ptr += AVX2_DOUBLE_STRIDE;
            y_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
    }

    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        x0 = _mm256_maskload_pd(x_ptr, mask);
        _mm256_maskstore_pd(y_ptr, mask, x0);
    }
}