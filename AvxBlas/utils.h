#pragma once

#pragma unmanaged

#include <immintrin.h>

static_assert(sizeof(float)  == 4, "sizeof float must be 4");
static_assert(sizeof(double) == 8, "sizeof float must be 8");

extern void alignment_vector_s(const unsigned int n, const unsigned int stride, const float* __restrict x_ptr, float* __restrict y_ptr);
extern void alignment_vector_d(const unsigned int n, const unsigned int stride, const double* __restrict x_ptr, double* __restrict y_ptr);

extern __m128i _mm_set_mask(const unsigned int n);
extern __m256i _mm256_set_mask(const unsigned int n);

union _m32 {
    float f;
    unsigned __int32 i;

    constexpr _m32(unsigned __int32 i) : i(i) { }
};

union _m64 {
    double f;
    unsigned __int64 i;

    constexpr _m64(unsigned __int64 i) : i(i) { }
};