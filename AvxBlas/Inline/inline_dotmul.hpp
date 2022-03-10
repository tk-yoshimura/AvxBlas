#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_sum.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline float dotmul_n32x_s(const unsigned int n, const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (n <= 0 || (n % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0, s1, s2, s3;
    unsigned int r = n;

    {
        s0 = _mm256_mul_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr));
        s1 = _mm256_mul_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE));
        s2 = _mm256_mul_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE * 2));
        s3 = _mm256_mul_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 3), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE * 3));

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    while (r >= AVX2_FLOAT_STRIDE * 4) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE), s1);
        s2 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE * 2), s2);
        s3 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 3), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE * 3), s3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

__forceinline float dotmul_alignment_s(unsigned int n, const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    unsigned int r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE), s1);
        s2 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE * 2), s2);
        s3 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 3), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE * 3), s3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r == AVX2_FLOAT_STRIDE * 3) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE), s1);
        s2 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE * 2), s2);
    }
    else if (r == AVX2_FLOAT_STRIDE * 2) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_load_ps(y_ptr + AVX2_FLOAT_STRIDE), s1);
    }
    else if (r == AVX2_FLOAT_STRIDE) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

__forceinline float dotmul_disorder_s(unsigned int n, const float* x_ptr, const float* y_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    unsigned int r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_loadu_ps(y_ptr + AVX2_FLOAT_STRIDE), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 2), _mm256_loadu_ps(y_ptr + AVX2_FLOAT_STRIDE * 2), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 3), _mm256_loadu_ps(y_ptr + AVX2_FLOAT_STRIDE * 3), s3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_loadu_ps(y_ptr + AVX2_FLOAT_STRIDE), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 2), _mm256_loadu_ps(y_ptr + AVX2_FLOAT_STRIDE * 2), s2);
        s3 = _mm256_fmadd_ps(_mm256_maskload_ps(x_ptr + AVX2_FLOAT_STRIDE * 3, mask), _mm256_maskload_ps(y_ptr + AVX2_FLOAT_STRIDE * 3, mask), s3);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE), _mm256_loadu_ps(y_ptr + AVX2_FLOAT_STRIDE), s1);
        s2 = _mm256_fmadd_ps(_mm256_maskload_ps(x_ptr + AVX2_FLOAT_STRIDE * 2, mask), _mm256_maskload_ps(y_ptr + AVX2_FLOAT_STRIDE * 2, mask), s2);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_maskload_ps(x_ptr + AVX2_FLOAT_STRIDE, mask), _mm256_maskload_ps(y_ptr + AVX2_FLOAT_STRIDE, mask), s1);
    }
    else {
        s0 = _mm256_fmadd_ps(_mm256_maskload_ps(x_ptr, mask), _mm256_maskload_ps(y_ptr, mask), s0);
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

__forceinline double dotmul_n16x_s(const unsigned int n, const double* x_ptr, const double* y_ptr) {
#ifdef _DEBUG
    if (n <= 0 || (n % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d s0, s1, s2, s3;
    unsigned int r = n;

    {
        s0 = _mm256_mul_pd(_mm256_load_pd(x_ptr), _mm256_load_pd(y_ptr));
        s1 = _mm256_mul_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE));
        s2 = _mm256_mul_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2));
        s3 = _mm256_mul_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 3), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3));

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        s0 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr), _mm256_load_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE), s1);
        s2 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2), s2);
        s3 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 3), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3), s3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }

    double ret = _mm256_sum16to1_pd(s0, s1, s2, s3);

    return ret;
}

__forceinline double dotmul_alignment_s(const unsigned int n, const double* x_ptr, const double* y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd(), s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
    unsigned int r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        s0 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr), _mm256_load_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE), s1);
        s2 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2), s2);
        s3 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 3), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3), s3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r == AVX2_DOUBLE_STRIDE * 3) {
        s0 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr), _mm256_load_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE), s1);
        s2 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2), s2);
    }
    else if (r == AVX2_DOUBLE_STRIDE * 2) {
        s0 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr), _mm256_load_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_load_pd(y_ptr + AVX2_DOUBLE_STRIDE), s1);
    }
    else if (r == AVX2_DOUBLE_STRIDE) {
        s0 = _mm256_fmadd_pd(_mm256_load_pd(x_ptr), _mm256_load_pd(y_ptr), s0);
    }

    double ret = _mm256_sum16to1_pd(s0, s1, s2, s3);

    return ret;
}

__forceinline double dotmul_disorder_s(const unsigned int n, const double* x_ptr, const double* y_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd(), s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
    unsigned int r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr), _mm256_loadu_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_loadu_pd(y_ptr + AVX2_DOUBLE_STRIDE), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2), _mm256_loadu_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2), s2);
        s3 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr + AVX2_DOUBLE_STRIDE * 3), _mm256_loadu_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3), s3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 3) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr), _mm256_loadu_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_loadu_pd(y_ptr + AVX2_DOUBLE_STRIDE), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2), _mm256_loadu_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2), s2);
        s3 = _mm256_fmadd_pd(_mm256_maskload_pd(x_ptr + AVX2_DOUBLE_STRIDE * 3, mask), _mm256_maskload_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3, mask), s3);
    }
    else if (r >= AVX2_DOUBLE_STRIDE * 2) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr), _mm256_loadu_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr + AVX2_DOUBLE_STRIDE), _mm256_loadu_pd(y_ptr + AVX2_DOUBLE_STRIDE), s1);
        s2 = _mm256_fmadd_pd(_mm256_maskload_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2, mask), _mm256_maskload_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2, mask), s2);
    }
    else if (r >= AVX2_DOUBLE_STRIDE) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(x_ptr), _mm256_loadu_pd(y_ptr), s0);
        s1 = _mm256_fmadd_pd(_mm256_maskload_pd(x_ptr + AVX2_DOUBLE_STRIDE, mask), _mm256_maskload_pd(y_ptr + AVX2_DOUBLE_STRIDE, mask), s1);
    }
    else {
        s0 = _mm256_fmadd_pd(_mm256_maskload_pd(x_ptr, mask), _mm256_maskload_pd(y_ptr, mask), s0);
    }

    double ret = _mm256_sum16to1_pd(s0, s1, s2, s3);

    return ret;
}