#pragma unmanaged

#include <immintrin.h>

__forceinline float _mm_sum4to1_ps(const __m128 x) {
    const __m128 y = _mm_add_ps(x, _mm_movehl_ps(x, x));
    const float ret = _mm_cvtss_f32(_mm_add_ss(y, _mm_shuffle_ps(y, y, 1)));

    return ret;
}

__forceinline float _mm256_sum8to1_ps(const __m256 x) {
    const __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 z = _mm_add_ps(y, _mm_movehl_ps(y, y));
    const float ret = _mm_cvtss_f32(_mm_add_ss(z, _mm_shuffle_ps(z, z, 1)));

    return ret;
}

__forceinline __m128 _mm256_sum8to2_ps(const __m256 x) {
    const __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 ret = _mm_add_ps(y, _mm_movehl_ps(y, y));

    return ret;
}

__forceinline __m128 _mm256_sum6to3_ps(const __m256 x) {
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

    const __m256 y = _mm256_permutevar8x32_ps(x, _perm43);
    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(y), _mm256_extractf128_ps(y, 1));

    return ret;
}

__forceinline __m128 _mm256_sum8to4_ps(const __m256 x) {
    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));

    return ret;
}

__forceinline double _mm256_sum4to1_pd(const __m256d x) {
    const __m128d y = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    const __m128d z = _mm_hadd_pd(y, y);
    const double ret = _mm_cvtsd_f64(z);

    return ret;
}

__forceinline __m128d _mm256_sum4to2_pd(const __m256d x) {
    const __m128d ret = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    
    return ret;
}