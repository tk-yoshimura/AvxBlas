#pragma once

#include <immintrin.h>
#include <chrono>
#include "../AvxBlas/constants.h"
#include "../AvxBlas/types.h"

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

int ag_sum_aligned_s_type1(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr);

int ag_sum_aligned_s_type2(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr);