#pragma once

#pragma unmanaged

#include <immintrin.h>

static_assert(sizeof(float)  == 4, "sizeof float must be 4");
static_assert(sizeof(double) == 8, "sizeof float must be 8");
static_assert(sizeof(unsigned int) == 4, "sizeof uint must be 4");

#define INPTR(type) const type* __restrict
#define OUTPTR(type) type* __restrict

typedef unsigned int uint;

extern void repeat_vector_s(const uint n, const uint stride, INPTR(float) x_ptr, OUTPTR(float) y_ptr);
extern void repeat_vector_d(const uint n, const uint stride, INPTR(double) x_ptr, OUTPTR(double) y_ptr);

extern __m128i _mm_setmask_ps(const uint n);
extern __m256i _mm256_setmask_ps(const uint n);
extern __m128i _mm_setmask_pd(const uint n);
extern __m256i _mm256_setmask_pd(const uint n);

extern void zeroset_s(const uint n, float* y_ptr);
extern void zeroset_d(const uint n, double* y_ptr);

extern void align_kernel_s(
    const uint n, const uint unaligned_w_size, const uint aligned_w_size,
    INPTR(float) unaligned_w_ptr, OUTPTR(float) aligned_w_ptr);
extern void align_kernel_d(
    const uint n, const uint unaligned_w_size, const uint aligned_w_size,
    INPTR(double) unaligned_w_ptr, OUTPTR(double) aligned_w_ptr);
extern void unalign_kernel_s(
    const uint n, const uint aligned_w_size, const uint unaligned_w_size,
    INPTR(float) aligned_w_ptr, OUTPTR(float) unaligned_w_ptr);
extern void unalign_kernel_d(
    const uint n, const uint aligned_w_size, const uint unaligned_w_size,
    INPTR(double) aligned_w_ptr, OUTPTR(double) unaligned_w_ptr);

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