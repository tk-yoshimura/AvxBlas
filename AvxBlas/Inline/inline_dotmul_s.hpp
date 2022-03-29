#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_sum_s.hpp"
#include "inline_loadstore_xn_s.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline float dotmul_n32x_s(const uint n, infloats x_ptr, infloats y_ptr) {
#ifdef _DEBUG
    if (n <= 0 || (n % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 s0, s1, s2, s3;

    uint r = n;

    {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        s0 = _mm256_mul_ps(x0, y0);
        s1 = _mm256_mul_ps(x1, y1);
        s2 = _mm256_mul_ps(x2, y2);
        s3 = _mm256_mul_ps(x3, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
        s3 = _mm256_fmadd_ps(x3, y3, s3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

__forceinline float dotmul_aligned_s(const uint n, infloats x_ptr, infloats y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
        s3 = _mm256_fmadd_ps(x3, y3, s3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);
        _mm256_load_x3_ps(y_ptr, y0, y1, y2);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);
        _mm256_load_x2_ps(y_ptr, y0, y1);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);
        _mm256_load_x1_ps(y_ptr, y0);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

__forceinline float dotmul_unaligned_s(const uint n, infloats x_ptr, infloats y_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
        s3 = _mm256_fmadd_ps(x3, y3, s3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);
        _mm256_maskload_x4_ps(y_ptr, y0, y1, y2, y3, mask);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
        s3 = _mm256_fmadd_ps(x3, y3, s3);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);
        _mm256_maskload_x3_ps(y_ptr, y0, y1, y2, mask);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
        s2 = _mm256_fmadd_ps(x2, y2, s2);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
        _mm256_maskload_x2_ps(y_ptr, y0, y1, mask);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
        s1 = _mm256_fmadd_ps(x1, y1, s1);
    }
    else {
        _mm256_maskload_x1_ps(x_ptr, x0, mask);
        _mm256_maskload_x1_ps(y_ptr, y0, mask);

        s0 = _mm256_fmadd_ps(x0, y0, s0);
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}