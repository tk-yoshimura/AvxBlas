#pragma unmanaged

#include <immintrin.h>

// e0,e1,e2,e3 -> e0+e1+e2+e3
__forceinline float _mm_sum4to1_ps(const __m128 x) {
    const __m128 y = _mm_add_ps(x, _mm_movehl_ps(x, x));
    const float ret = _mm_cvtss_f32(_mm_add_ss(y, _mm_shuffle_ps(y, y, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1+e2+e3+e4+e5+e6+e7
__forceinline float _mm256_sum8to1_ps(const __m256 x) {
    const __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 z = _mm_add_ps(y, _mm_movehl_ps(y, y));
    const float ret = _mm_cvtss_f32(_mm_add_ss(z, _mm_shuffle_ps(z, z, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e2+e4+e6,e1+e3+e5+e7
__forceinline __m128 _mm256_sum8to2_ps(const __m256 x) {
    const __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 ret = _mm_add_ps(y, _mm_movehl_ps(y, y));

    return ret;
}

// e0,e1,e2,e3,e4,e5,-,- -> e0+e3,e1+e4,e2+e5,-
__forceinline __m128 _mm256_sum6to3_ps(const __m256 x) {
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

    const __m256 y = _mm256_permutevar8x32_ps(x, _perm43);
    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(y), _mm256_extractf128_ps(y, 1));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e4,e1+e5,e2+e6,e3+e7
__forceinline __m128 _mm256_sum8to4_ps(const __m256 x) {
    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));

    return ret;
}

// e0,...,e15 -> e0+...+e15
__forceinline float _mm256_sum16to1_ps(const __m256 x, const __m256 y) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(x, y));

    return ret;
}

// e0,...,e23 -> e0+...+e23
__forceinline float _mm256_sum24to1_ps(const __m256 x, const __m256 y, const __m256 z) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(_mm256_add_ps(x, y), z));

    return ret;
}

// e0,...,e31 -> e0+...+e31
__forceinline float _mm256_sum32to1_ps(const __m256 x, const __m256 y, const __m256 z, const __m256 w) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(_mm256_add_ps(x, y), _mm256_add_ps(z, w)));

    return ret;
}

// e0,e1 -> e0+e1
__forceinline double _mm_sum2to1_pd(const __m128d x) {
    const __m128d y = _mm_hadd_pd(x, x);
    const double ret = _mm_cvtsd_f64(y);

    return ret;
}

// e0,e1,e2,e3 -> e0+e1+e2+e3
__forceinline double _mm256_sum4to1_pd(const __m256d x) {
    const __m128d y = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    const __m128d z = _mm_hadd_pd(y, y);
    const double ret = _mm_cvtsd_f64(z);

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e2+e4+e6,e1+e3+e5+e7
__forceinline __m128d _mm256_sum4to2_pd(const __m256d x) {
    const __m128d ret = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    
    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e2+e4+e6,e1+e3+e5+e7
__forceinline double _mm256_sum8to1_ps(const __m256d x, const __m256d y) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(x, y));

    return ret;
}

// e0,...,e15 -> e0+...+e15
__forceinline double _mm256_sum16to1_ps(const __m256d x, const __m256d y, const __m256d z, const __m256d w) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(_mm256_add_pd(x, y), _mm256_add_pd(z, w)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1,e2+e3,e4+e5,e6+e7,-,-,-,-
__forceinline __m256 _mm256_hadd2_ps(const __m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    const __m256 y = _mm256_hadd_ps(x, x);
    const __m256 ret = _mm256_permutevar8x32_ps(y, _perm);

    return ret;
}

// e0,e1,e2,e3,e4,e5,_,_ -> e0+e1+e2,e3+e4+e5,-,-,-,-,-,-
__forceinline __m256 _mm256_hadd3_ps(const __m256 x) {
    const __m256i _perm82 = _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7);
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);
    const __m256 _mask43 = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, ~0u, ~0u, 0, ~0u, ~0u, ~0u, 0));

    const __m256 y = _mm256_and_ps(_mm256_permutevar8x32_ps(x, _perm43), _mask43);
    const __m256 z = _mm256_hadd_ps(y, y);
    const __m256 w = _mm256_hadd_ps(z, z);
    const __m256 ret = _mm256_permutevar8x32_ps(w, _perm82);

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1+e2+e3,e4+e5+e6+e7,-,-,-,-,-,-
__forceinline __m256 _mm256_hadd4_ps(const __m256 x) {
    const __m256i _perm82 = _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7);

    const __m256 y = _mm256_hadd_ps(x, x);
    const __m256 z = _mm256_hadd_ps(y, y);
    const __m256 ret = _mm256_permutevar8x32_ps(z, _perm82);

    return ret;
}