#pragma once
#pragma unmanaged
#pragma warning(error: 4714)

#include <immintrin.h>

extern void alignment_vector_s(const unsigned int n, const unsigned int stride, const float* __restrict x_ptr, float* __restrict y_ptr);
extern void alignment_vector_d(const unsigned int n, const unsigned int stride, const double* __restrict x_ptr, double* __restrict y_ptr);

extern __m128i _mm_set_mask(const unsigned int n);
extern __m256i _mm256_set_mask(const unsigned int n);

extern __forceinline float _mm256_sum8to1_ps(const __m256 x);
extern __forceinline __m128 _mm256_sum8to2_ps(const __m256 x);
extern __forceinline __m128 _mm256_sum6to3_ps(const __m256 x);
extern __forceinline __m128 _mm256_sum8to4_ps(const __m256 x);

#define SUCCESS (0)
#define FAILURE_BADPARAM (-1)
#define FAILURE_BADALLOC (-2)

#define AVX2_ALIGNMENT (32)

#define AVX2_FLOAT_STRIDE (8)
#define AVX2_FLOAT_BATCH_MASK (~7u)
#define AVX2_FLOAT_REMAIN_MASK (7u)

#define AVX2_DOUBLE_STRIDE (4)
#define AVX2_DOUBLE_BATCH_MASK (~3u)
#define AVX2_DOUBLE_REMAIN_MASK (3u)

#define MAX_VECTORWISE_ALIGNMNET_INCX (4096)
#define MAX_AGGREGATE_BATCHING (64)

#pragma managed