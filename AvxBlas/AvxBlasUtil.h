#pragma once

#pragma warning(disable: 4793)

namespace AvxBlas {
    extern unsigned int gcd(unsigned int a, unsigned int b);
    extern unsigned int lcm(unsigned int a, unsigned int b);
    extern unsigned long gcd(unsigned long a, unsigned long b);
    extern unsigned long lcm(unsigned long a, unsigned long b);
    extern void alignment_vector_s(unsigned int n, unsigned int incx, const float* __restrict x_ptr, float* __restrict y_ptr);
    extern void alignment_vector_d(unsigned int n, unsigned int incx, const double* __restrict x_ptr, double* __restrict y_ptr);
}

#define AVX2_ALIGNMENT (32)

#define AVX2_FLOAT_STRIDE (8)
#define AVX2_FLOAT_BATCH_MASK (~7u)
#define AVX2_FLOAT_REMAIN_MASK (7u)

#define AVX2_DOUBLE_STRIDE (4)
#define AVX2_DOUBLE_BATCH_MASK (~3u)
#define AVX2_DOUBLE_REMAIN_MASK (3u)

#define MAX_VECTORWISE_ALIGNMNET_INCX (4096)
#define MAX_VECTORWISE_ALIGNMNET_ULENGTH (4096)

#define mm256_mask(ret, k) \
    const int __mask_v[15] = { -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 }; \
    const int __mask_j = 7 - (k); \
    ret = _mm256_setr_epi32(__mask_v[__mask_j], __mask_v[__mask_j + 1], __mask_v[__mask_j + 2], __mask_v[__mask_j + 3], __mask_v[__mask_j + 4], __mask_v[__mask_j + 5], __mask_v[__mask_j + 6], __mask_v[__mask_j + 7]);

#define mm128_mask(ret, k) \
    const int __mask_v[7] = { -1, -1, -1, 0, 0, 0, 0 }; \
    const int __mask_j = 3 - (k); \
    ret = _mm_setr_epi32(__mask_v[__mask_j], __mask_v[__mask_j + 1], __mask_v[__mask_j + 2], __mask_v[__mask_j + 3]);