#pragma once
#pragma warning(disable: 4793)

#include <immintrin.h>

extern unsigned int gcd(unsigned int a, unsigned int b);
extern unsigned int lcm(unsigned int a, unsigned int b);
extern unsigned long gcd(unsigned long a, unsigned long b);
extern unsigned long lcm(unsigned long a, unsigned long b);
extern void alignment_vector_s(unsigned int n, unsigned int stride, const float* __restrict x_ptr, float* __restrict y_ptr);
extern void alignment_vector_d(unsigned int n, unsigned int stride, const double* __restrict x_ptr, double* __restrict y_ptr);
extern __m256i mm256_mask(unsigned int n);
extern __m128i mm128_mask(unsigned int n);

#define AVX2_ALIGNMENT (32)

#define AVX2_FLOAT_STRIDE (8)
#define AVX2_FLOAT_BATCH_MASK (~7u)
#define AVX2_FLOAT_REMAIN_MASK (7u)

#define AVX2_DOUBLE_STRIDE (4)
#define AVX2_DOUBLE_BATCH_MASK (~3u)
#define AVX2_DOUBLE_REMAIN_MASK (3u)

//#define AVX1_ALIGNMENT (16)
//
//#define AVX1_FLOAT_STRIDE (4)
//#define AVX1_FLOAT_BATCH_MASK (~3u)
//#define AVX1_FLOAT_REMAIN_MASK (3u)
//
//#define AVX1_DOUBLE_STRIDE (2)
//#define AVX1_DOUBLE_BATCH_MASK (~1u)
//#define AVX1_DOUBLE_REMAIN_MASK (1u)

#define MAX_VECTORWISE_ALIGNMNET_INCX (4096)
#define MAX_VECTORWISE_ALIGNMNET_ULENGTH (4096)
#define MAX_AGGREGATE_BATCHING (64)