#include "../avxblas_sandbox.h"

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1+e2+e3+e4+e5+e6+e7
__forceinline float _mm256_sum8to1_ps(const __m256 x) {
    const __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 z = _mm_add_ps(y, _mm_movehl_ps(y, y));
    const float ret = _mm_cvtss_f32(_mm_add_ss(z, _mm_shuffle_ps(z, z, 1)));

    return ret;
}

// e0,...,e15 -> e0+...+e15
__forceinline float _mm256_sum16to1_ps(const __m256 x, const __m256 y) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(x, y));

    return ret;
}

// e0,...,e31 -> e0+...+e31
__forceinline float _mm256_sum32to1_ps(const __m256 x, const __m256 y, const __m256 z, const __m256 w) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(_mm256_add_ps(x, y), _mm256_add_ps(z, w)));

    return ret;
}

float dotmul_stride8_s(
    unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr) {

    __m256 s0 = _mm256_setzero_ps();

    while (n >= AVX2_FLOAT_STRIDE) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        s0 = _mm256_fmadd_ps(x01, x02, s0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        n -= AVX2_FLOAT_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_setmask_ps(n);

        __m256 x1 = _mm256_maskload_ps(x1_ptr, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr, mask);

        s0 = _mm256_fmadd_ps(x1, x2, s0);
    }

    float ret = _mm256_sum8to1_ps(s0);

    return ret;
}

float dotmul_stride16_s(
    unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr) {

    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();

    while (n >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        __m256 x11 = _mm256_load_ps(x1_ptr + 8);
        __m256 x12 = _mm256_load_ps(x2_ptr + 8);

        s0 = _mm256_fmadd_ps(x01, x02, s0);
        s1 = _mm256_fmadd_ps(x11, x12, s1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        n -= AVX2_FLOAT_STRIDE * 2;
    }
    if (n >= AVX2_FLOAT_STRIDE) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        s0 = _mm256_fmadd_ps(x01, x02, s0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        n -= AVX2_FLOAT_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_setmask_ps(n);

        __m256 x1 = _mm256_maskload_ps(x1_ptr, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr, mask);

        s1 = _mm256_fmadd_ps(x1, x2, s1);
    }

    float ret = _mm256_sum16to1_ps(s0, s1);

    return ret;
}

float dotmul_stride32_s(
    unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr) {

    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

    while(n >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        __m256 x11 = _mm256_load_ps(x1_ptr + 8);
        __m256 x12 = _mm256_load_ps(x2_ptr + 8);
        
        __m256 x21 = _mm256_load_ps(x1_ptr + 16);
        __m256 x22 = _mm256_load_ps(x2_ptr + 16);
        
        __m256 x31 = _mm256_load_ps(x1_ptr + 24);
        __m256 x32 = _mm256_load_ps(x2_ptr + 24);

        s0 = _mm256_fmadd_ps(x01, x02, s0);
        s1 = _mm256_fmadd_ps(x11, x12, s1);
        s2 = _mm256_fmadd_ps(x21, x22, s2);
        s3 = _mm256_fmadd_ps(x31, x32, s3);

        x1_ptr += AVX2_FLOAT_STRIDE * 4;
        x2_ptr += AVX2_FLOAT_STRIDE * 4;
        n -= AVX2_FLOAT_STRIDE * 4;
    }
    if(n >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        __m256 x11 = _mm256_load_ps(x1_ptr + 8);
        __m256 x12 = _mm256_load_ps(x2_ptr + 8);

        s0 = _mm256_fmadd_ps(x01, x02, s0);
        s1 = _mm256_fmadd_ps(x11, x12, s1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        n -= AVX2_FLOAT_STRIDE * 2;
    }
    if (n >= AVX2_FLOAT_STRIDE) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        s2 = _mm256_fmadd_ps(x01, x02, s2);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        n -= AVX2_FLOAT_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_setmask_ps(n);

        __m256 x1 = _mm256_maskload_ps(x1_ptr, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr, mask);

        s3 = _mm256_fmadd_ps(x1, x2, s3);
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}