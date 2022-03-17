#pragma once

#pragma unmanaged

#include <immintrin.h>

static_assert(sizeof(float)  == 4, "sizeof float must be 4");
static_assert(sizeof(double) == 8, "sizeof float must be 8");

extern void repeat_vector_s(const unsigned int n, const unsigned int stride, const float* __restrict x_ptr, float* __restrict y_ptr);
extern void repeat_vector_d(const unsigned int n, const unsigned int stride, const double* __restrict x_ptr, double* __restrict y_ptr);

extern __m128i _mm_setmask_ps(const unsigned int n);
extern __m256i _mm256_setmask_ps(const unsigned int n);
extern __m128i _mm_setmask_pd(const unsigned int n);
extern __m256i _mm256_setmask_pd(const unsigned int n);

extern void zeroset_s(const unsigned int n, float* y_ptr);
extern void zeroset_d(const unsigned int n, double* y_ptr);

extern void align_kernel_s(
    const unsigned int n, const unsigned int unaligned_w_size, const unsigned int aligned_w_size,
    const float* __restrict unaligned_w_ptr, float* __restrict aligned_w_ptr);
extern void align_kernel_d(
    const unsigned int n, const unsigned int unaligned_w_size, const unsigned int aligned_w_size,
    const double* __restrict unaligned_w_ptr, double* __restrict aligned_w_ptr);

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