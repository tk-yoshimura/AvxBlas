#pragma once

#pragma warning(disable: 4793)

#include <immintrin.h>

namespace AvxBlas {
    extern __m256i masktable_m256(unsigned int k);
    extern __m128i masktable_m128(unsigned int k);
    extern unsigned int gcd(unsigned int a, unsigned int b);
    extern unsigned int lcm(unsigned int a, unsigned int b);
    extern unsigned long gcd(unsigned long a, unsigned long b);
    extern unsigned long lcm(unsigned long a, unsigned long b);
    extern void alignment_vector_s(unsigned int n, unsigned int incx, const float* __restrict x_ptr, float* __restrict y_ptr);
    extern void alignment_vector_d(unsigned int n, unsigned int incx, const double* __restrict x_ptr, double* __restrict y_ptr);
}