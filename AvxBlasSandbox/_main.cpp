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

// e0,...,e39 -> e0+e5+...+e35,e1+e6+...+e36,e2+e7+...+e37,e3+e8+...+e38,e4+e9+...+e39,zero,zero,zero
__forceinline __m256 _mm256_sum40to5_ps(const __m256 x0, const __m256 x1, const __m256 x2, const __m256 x3, const __m256 x4) {

    const __m256i _perm_y1 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256i _perm_y2 = _mm256_setr_epi32(4, 5, 6, 7, 3, 0, 1, 2);
    const __m256i _perm_y3 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i _perm_y4 = _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2);

    const __m256i _perm_z0 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);
    const __m256i _perm_z1 = _mm256_setr_epi32(5, 0, 1, 6, 7, 2, 3, 4);
    const __m256i _perm_z2 = _mm256_setr_epi32(0, 5, 6, 7, 1, 2, 3, 4);
    const __m256i _perm_z3 = _mm256_setr_epi32(5, 6, 0, 1, 7, 2, 3, 4);
    const __m256i _perm_z4 = _mm256_setr_epi32(0, 1, 5, 6, 7, 2, 3, 4);

    const __m256 _mask_1 = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, 0, 0, 0));
    const __m256 _mask_2 = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, 0, 0, ~0u, ~0u, ~0u));

    const __m256 y0 = x0;
    const __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    const __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);
    const __m256 y3 = _mm256_permutevar8x32_ps(x3, _perm_y3);
    const __m256 y4 = _mm256_permutevar8x32_ps(x4, _perm_y4);

    const __m256 z0 = _mm256_permutevar8x32_ps(_mm256_and_ps(y0, _mask_2), _perm_z0);
    const __m256 z1 = _mm256_permutevar8x32_ps(_mm256_and_ps(y1, _mask_2), _perm_z1);
    const __m256 z2 = _mm256_permutevar8x32_ps(_mm256_and_ps(y2, _mask_2), _perm_z2);
    const __m256 z3 = _mm256_permutevar8x32_ps(_mm256_and_ps(y3, _mask_2), _perm_z3);
    const __m256 z4 = _mm256_permutevar8x32_ps(_mm256_and_ps(y4, _mask_2), _perm_z4);

    const __m256 w0 = _mm256_and_ps(_mm256_add_ps(y0, _mm256_add_ps(_mm256_add_ps(y1, y2), _mm256_add_ps(y3, y4))), _mask_1);
    const __m256 w1 = _mm256_add_ps(z0, _mm256_add_ps(_mm256_add_ps(z1, z2), _mm256_add_ps(z3, z4)));

    const __m256 s = _mm256_add_ps(w0, w1);

    return s;
}

// e0,...,e19 -> e0+e5+...+e15,e1+e6+...+e16,e2+e7+...+e17,e3+e8+...+e18,e4+e9+...+e19,_,_,_
__forceinline __m256dx2 _mm256_sum20to5_pd(const __m256d x0, const __m256d x1, const __m256d x2, const __m256d x3, const __m256d x4) {
    const __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ADCB);
    const __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABDC);
    const __m256d y3 = _mm256_permute4x64_pd(x3, _MM_PERM_BACD);
    const __m256d y4 = _mm256_permute4x64_pd(x4, _MM_PERM_CBAD);

    const __m256d z0 = x0;
    const __m256d z1 = _mm256_blend_pd(y1, y4, 0b1000);
    const __m256d z2 = _mm256_blend_pd(y2, y4, 0b0100);
    const __m256d z3 = _mm256_blend_pd(y3, y4, 0b0010);
    
    const __m256d w0 = _mm256_blend_pd(y1, y2, 0b0100);
    const __m256d w1 = _mm256_blend_pd(y3, y4, 0b0001);
    const __m256d wc = _mm256_blend_pd(w0, w1, 0b0011);
    const __m256d wa = _mm256_permute4x64_pd(_mm256_hadd_pd(wc, wc), _MM_PERM_DBCA);
    const __m256d wb = _mm256_hadd_pd(wa, wa);

    const __m256d lo = _mm256_add_pd(_mm256_add_pd(z0, z1), _mm256_add_pd(z2, z3));
    const __m256d hi = wb;

    __m256dx2 ret(lo, hi);

    return ret;
}

int main(){
    __m256d x0 = _mm256_setr_pd(11, 12, 13, 14);
    __m256d x1 = _mm256_setr_pd(15, 21, 22, 23);
    __m256d x2 = _mm256_setr_pd(24, 25, 31, 32);
    __m256d x3 = _mm256_setr_pd(33, 34, 35, 41);
    __m256d x4 = _mm256_setr_pd(42, 43, 44, 45);

    __m256dx2 ret = _mm256_sum20to5_pd(x0, x1, x2, x3, x4);

    getchar();
}
