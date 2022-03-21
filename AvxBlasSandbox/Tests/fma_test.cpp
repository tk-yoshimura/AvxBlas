#include <immintrin.h>

// (s, c) += a * b
__forceinline void _mm256_fasttwosum_fmadd_ps(const __m256 a, const __m256 b, __m256& s, __m256& c) {
    __m256 tmp = s;
    s = _mm256_fmadd_ps(a, b, s);
    c = _mm256_add_ps(c, _mm256_fmadd_ps(a, b, _mm256_sub_ps(tmp, s)));
}

// (s, c) += a * b
__forceinline void _mm256_twosum_fmadd_ps(const __m256 a, const __m256 b, __m256& s, __m256& c) {
    __m256 tmp = s;
    s = _mm256_fmadd_ps(a, b, _mm256_add_ps(c, s));
    c = _mm256_add_ps(c, _mm256_fmadd_ps(a, b, _mm256_sub_ps(tmp, s)));
}