#include "../avxblasutil.h"

#pragma unmanaged

__forceinline __m128 _mm256_sum8to1_ps(__m256 x) {
    __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    __m128 z = _mm_add_ps(y, _mm_movehl_ps(y, y));
    __m128 ret = _mm_add_ss(z, _mm_shuffle_ps(z, z, 1));

    return ret;
}

__forceinline __m128 _mm256_sum8to2_ps(__m256 x) {
    __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    __m128 ret = _mm_add_ps(y, _mm_movehl_ps(y, y));

    return ret;
}

__forceinline __m128 _mm256_sum6to3_ps(__m256 x) {
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, _perm43);
    __m128 ret = _mm_add_ps(_mm256_castps256_ps128(y), _mm256_extractf128_ps(y, 1));

    return ret;
}

__forceinline __m128 _mm256_sum8to4_ps(__m256 x) {
    __m128 ret = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));

    return ret;
}
