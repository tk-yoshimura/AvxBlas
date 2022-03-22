#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"

// e0,...,e23 -> e0+e3+...+e21,e1+e4+...+e22,e2+e5+...+e23,zero
__forceinline __m128 _mm256_sum24to3_ps(const __m256 x0, const __m256 x1, const __m256 x2) {
    const __m256i _perm_y0 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);
    const __m256i _perm_y1 = _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7);
    const __m256i _perm_y2 = _mm256_setr_epi32(2, 3, 4, 1, 5, 6, 7, 0);

    const __m256i _perm_z0 = _mm256_setr_epi32(3, 7, 0, 1, 2, 4, 5, 6);
    const __m256i _perm_z1 = _mm256_setr_epi32(7, 0, 3, 1, 2, 4, 5, 6);
    const __m256i _perm_z2 = _mm256_setr_epi32(0, 7, 3, 1, 2, 4, 5, 6);

    const __m256 _mask_1 = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, ~0u, ~0u, 0, ~0u, ~0u, ~0u, 0));
    const __m256 _mask_2 = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, ~0u, 0, 0, 0, ~0u));

    const __m256 y0 = _mm256_permutevar8x32_ps(x0, _perm_y0);
    const __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    const __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);

    const __m256 z0 = _mm256_permutevar8x32_ps(_mm256_and_ps(y0, _mask_2), _perm_z0);
    const __m256 z1 = _mm256_permutevar8x32_ps(_mm256_and_ps(y1, _mask_2), _perm_z1);
    const __m256 z2 = _mm256_permutevar8x32_ps(_mm256_and_ps(y2, _mask_2), _perm_z2);

    const __m256 w0 = _mm256_and_ps(_mm256_add_ps(y0, _mm256_add_ps(y1, y2)), _mask_1);
    const __m256 w1 = _mm256_add_ps(z0, _mm256_add_ps(z1, z2));

    const __m256 s = _mm256_add_ps(w0, w1);

    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));

    return ret;
}

// e0,...,e11 -> e0+e3+...+e9,e1+e4+...+e10,e2+e5+...+e11,zero
__forceinline __m256d _mm256_sum12to3_pd(const __m256d x0, const __m256d x1, const __m256d x2) {
    const __m256d _mask_1 = _mm256_castsi256_pd(_mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0));
    const __m256d _mask_2 = _mm256_castsi256_pd(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, ~0u, ~0u));

    const __m256d y0 = x0;
    const __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABDC);
    const __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ADCB);

    const __m256d z0 = _mm256_permute4x64_pd(_mm256_and_pd(y0, _mask_2), _MM_PERM_AAAD);
    const __m256d z1 = _mm256_permute4x64_pd(_mm256_and_pd(y1, _mask_2), _MM_PERM_AADA);
    const __m256d z2 = _mm256_permute4x64_pd(_mm256_and_pd(y2, _mask_2), _MM_PERM_ADAA);

    const __m256d w0 = _mm256_and_pd(_mm256_add_pd(y0, _mm256_add_pd(y1, y2)), _mask_1);
    const __m256d w1 = _mm256_add_pd(z0, _mm256_add_pd(z1, z2));

    const __m256d ret = _mm256_add_pd(w0, w1);

    return ret;
}

// e0,...,e23 -> e0+e6+...+e18,e1+e7+...+e19,e2+e8+...+e20,e3+e9+...+e21,e4+e10+...+e22,e5+e11+...+e23,zero,zero
__forceinline __m256 _mm256_sum24to6_ps(const __m256 x0, const __m256 x1, const __m256 x2) {
    const __m256i _perm_y0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i _perm_y1 = _mm256_setr_epi32(4, 5, 6, 7, 2, 3, 0, 1);
    const __m256i _perm_y2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);

    const __m256i _perm_z0 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
    const __m256i _perm_z1 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    const __m256i _perm_z2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);

    const __m256 _mask_1 = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0));
    const __m256 _mask_2 = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, ~0u, ~0u));

    const __m256 y0 = _mm256_permutevar8x32_ps(x0, _perm_y0);
    const __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    const __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);

    const __m256 z0 = _mm256_permutevar8x32_ps(_mm256_and_ps(y0, _mask_2), _perm_z0);
    const __m256 z1 = _mm256_permutevar8x32_ps(_mm256_and_ps(y1, _mask_2), _perm_z1);
    const __m256 z2 = _mm256_permutevar8x32_ps(_mm256_and_ps(y2, _mask_2), _perm_z2);

    const __m256 w0 = _mm256_and_ps(_mm256_add_ps(y0, _mm256_add_ps(y1, y2)), _mask_1);
    const __m256 w1 = _mm256_add_ps(z0, _mm256_add_ps(z1, z2));

    const __m256 s = _mm256_add_ps(w0, w1);

    return s;
}

int main(){
    {
        __m256 x0 = _mm256_setr_ps(11, 12, 13, 14, 15, 16, 21, 22);
        __m256 x1 = _mm256_setr_ps(23, 24, 25, 26, 31, 32, 33, 34);
        __m256 x2 = _mm256_setr_ps(35, 36, 41, 42, 43, 44, 45, 46);

        __m256 y = _mm256_sum24to6_ps(x0, x1, x2);
    }

    {
        __m256d x0 = _mm256_setr_pd(1, 2, 3, 11);
        __m256d x1 = _mm256_setr_pd(12, 13, 21, 22);
        __m256d x2 = _mm256_setr_pd(23, 31, 32, 33);

        __m256d y = _mm256_sum12to3_pd(x0, x1, x2);
    }

    getchar();
}
