#pragma once

#include <immintrin.h>
#include <chrono>
#include "../AvxBlas/constants.h"

extern __m256i _mm256_setmask_ps(const unsigned int n);
extern __m128i _mm_setmask_ps(const unsigned int n);
extern __m256i _mm256_setmask_pd(const unsigned int n);
extern __m128i _mm_setmask_pd(const unsigned int n);

extern int add_stride8_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr);

extern int add_stride16_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr);

extern int add_stride32_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr);

extern float dotmul_stride8_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr);

extern float dotmul_stride16_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr);

extern float dotmul_stride32_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr);

struct __m256dx2 {
    __m256d lo, hi;

    constexpr __m256dx2(__m256d lo, __m256d hi) : lo(lo), hi(hi) { }
};