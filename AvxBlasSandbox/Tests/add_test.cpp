#include "../avxblas_sandbox.h"

int add_stride8_s(
    unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr) {

    while (n >= AVX2_FLOAT_STRIDE) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        __m256 y0 = _mm256_add_ps(x01, x02);

        _mm256_stream_ps(y_ptr, y0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        n -= AVX2_FLOAT_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_set_mask(n);

        __m256 x1 = _mm256_maskload_ps(x1_ptr, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr, mask);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_maskstore_ps(y_ptr, mask, y);
    }

    return 0;
}

int add_stride16_s(
    unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr) {

    while (n >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);
        __m256 y0 = _mm256_add_ps(x01, x02);

        __m256 x11 = _mm256_load_ps(x1_ptr + 8);
        __m256 x12 = _mm256_load_ps(x2_ptr + 8);
        __m256 y1 = _mm256_add_ps(x11, x12);

        _mm256_stream_ps(y_ptr, y0);
        _mm256_stream_ps(y_ptr + 8, y1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        n -= AVX2_FLOAT_STRIDE * 2;
    }
    if (n >= AVX2_FLOAT_STRIDE) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        __m256 y0 = _mm256_add_ps(x01, x02);

        _mm256_stream_ps(y_ptr, y0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        n -= AVX2_FLOAT_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_set_mask(n);

        __m256 x1 = _mm256_maskload_ps(x1_ptr, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr, mask);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_maskstore_ps(y_ptr, mask, y);
    }

    return 0;
}

int add_stride32_s(
    unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr) {

    while(n >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);
        __m256 y0 = _mm256_add_ps(x01, x02);

        __m256 x11 = _mm256_load_ps(x1_ptr + 8);
        __m256 x12 = _mm256_load_ps(x2_ptr + 8);
        __m256 y1 = _mm256_add_ps(x11, x12);
        
        __m256 x21 = _mm256_load_ps(x1_ptr + 16);
        __m256 x22 = _mm256_load_ps(x2_ptr + 16);
        __m256 y2 = _mm256_add_ps(x21, x22);
        
        __m256 x31 = _mm256_load_ps(x1_ptr + 24);
        __m256 x32 = _mm256_load_ps(x2_ptr + 24);
        __m256 y3 = _mm256_add_ps(x31, x32);

        _mm256_stream_ps(y_ptr, y0);
        _mm256_stream_ps(y_ptr + 8, y1);
        _mm256_stream_ps(y_ptr + 16, y2);
        _mm256_stream_ps(y_ptr + 24, y3);

        x1_ptr += AVX2_FLOAT_STRIDE * 4;
        x2_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr  += AVX2_FLOAT_STRIDE * 4;
        n -= AVX2_FLOAT_STRIDE * 4;
    }
    if(n >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);
        __m256 x11 = _mm256_load_ps(x1_ptr + 8);
        __m256 x12 = _mm256_load_ps(x2_ptr + 8);

        __m256 y0 = _mm256_add_ps(x01, x02);
        __m256 y1 = _mm256_add_ps(x11, x12);

        _mm256_stream_ps(y_ptr, y0);
        _mm256_stream_ps(y_ptr + 8, y1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        n -= AVX2_FLOAT_STRIDE * 2;
    }
    if (n >= AVX2_FLOAT_STRIDE) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        __m256 y0 = _mm256_add_ps(x01, x02);

        _mm256_stream_ps(y_ptr, y0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        n -= AVX2_FLOAT_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_set_mask(n);

        __m256 x1 = _mm256_maskload_ps(x1_ptr, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr, mask);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_maskstore_ps(y_ptr, mask, y);
    }

    return 0;
}