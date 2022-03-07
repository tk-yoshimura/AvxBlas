#pragma once

#pragma unmanaged

#include <immintrin.h>

extern void alignment_vector_s(const unsigned int n, const unsigned int stride, const float* __restrict x_ptr, float* __restrict y_ptr);
extern void alignment_vector_d(const unsigned int n, const unsigned int stride, const double* __restrict x_ptr, double* __restrict y_ptr);

extern __m128i _mm_set_mask(const unsigned int n);
extern __m256i _mm256_set_mask(const unsigned int n);