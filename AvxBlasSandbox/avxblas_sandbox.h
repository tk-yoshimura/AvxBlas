#include <immintrin.h>
#include <chrono>

#define AVX2_ALIGNMENT (32)
#define AVX2_FLOAT_STRIDE (8)
#define AVX2_FLOAT_BATCH_MASK (~7u)
#define AVX2_FLOAT_REMAIN_MASK (7u)

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