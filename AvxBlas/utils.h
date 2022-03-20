#pragma once

#pragma unmanaged

#include <immintrin.h>
#include "types.h"

extern void repeat_vector_s(const uint n, const uint stride, infloats x_ptr, outfloats y_ptr);
extern void repeat_vector_d(const uint n, const uint stride, indoubles x_ptr, outdoubles y_ptr);

extern __m128i _mm_setmask_ps(const uint n);
extern __m256i _mm256_setmask_ps(const uint n);
extern __m128i _mm_setmask_pd(const uint n);
extern __m256i _mm256_setmask_pd(const uint n);

extern void zeroset_s(const uint n, outfloats y_ptr);
extern void zeroset_d(const uint n, outdoubles y_ptr);

extern void align_kernel_s(
    const uint n, const uint unaligned_w_size, const uint aligned_w_size,
    infloats unaligned_w_ptr, outfloats aligned_w_ptr);
extern void align_kernel_d(
    const uint n, const uint unaligned_w_size, const uint aligned_w_size,
    indoubles unaligned_w_ptr, outdoubles aligned_w_ptr);
extern void unalign_kernel_s(
    const uint n, const uint aligned_w_size, const uint unaligned_w_size,
    infloats aligned_w_ptr, outfloats unaligned_w_ptr);
extern void unalign_kernel_d(
    const uint n, const uint aligned_w_size, const uint unaligned_w_size,
    indoubles aligned_w_ptr, outdoubles unaligned_w_ptr);

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