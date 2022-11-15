#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "../AvxBlas/Inline/inline_max_d.hpp"
#include "../AvxBlas/types.h"
#include "../AvxBlas/Util/util_mm_mask.cpp"

// e0,...,e19 -> e0+e5+...+e15,e1+e6+...+e16,e2+e7+...+e17,e3+e8+...+e18,e4+e9+...+e19,_,_,_
__forceinline __m256dx2 _mm256_sum20to5_pd_r1(__m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4) {
    __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ADCB);
    __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABDC);
    __m256d y3 = _mm256_permute4x64_pd(x3, _MM_PERM_BACD);
    __m256d y4 = _mm256_permute4x64_pd(x4, _MM_PERM_CBAD);

    __m256d z0 = x0;
    __m256d z1 = _mm256_blend_pd(y1, y4, 0b1000);
    __m256d z2 = _mm256_blend_pd(y2, y4, 0b0100);
    __m256d z3 = _mm256_blend_pd(y3, y4, 0b0010);

    __m256d w0 = _mm256_blend_pd(y1, y2, 0b0100);
    __m256d w1 = _mm256_blend_pd(y3, y4, 0b0001);
    __m256d wc = _mm256_blend_pd(w0, w1, 0b0011);
    __m256d wa = _mm256_permute4x64_pd(_mm256_hadd_pd(wc, wc), _MM_PERM_DBCA);
    __m256d wb = _mm256_hadd_pd(wa, wa);

    __m256d imm0 = _mm256_add_pd(_mm256_add_pd(z0, z1), _mm256_add_pd(z2, z3));
    __m256d imm1 = wb;

    return __m256dx2(imm0, imm1);
}

__forceinline __m256dx2 _mm256_sum20to5_pd_r2(__m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4) {
    __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ADCB);
    __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABDC);
    __m256d y3 = _mm256_permute4x64_pd(x3, _MM_PERM_BACD);
    __m256d y4 = _mm256_permute4x64_pd(x4, _MM_PERM_CBAD);

    __m256d z0 = x0;
    __m256d z1 = _mm256_blend_pd(y1, y4, 0b1000);
    __m256d z2 = _mm256_blend_pd(y2, y4, 0b0100);
    __m256d z3 = _mm256_blend_pd(y3, y4, 0b0010);

    __m256d w0 = _mm256_blend_pd(y1, y2, 0b0100);
    __m256d w1 = _mm256_blend_pd(y3, y4, 0b0001);
    __m256d wc = _mm256_blend_pd(w0, w1, 0b0011);
    __m256d wa = _mm256_permute4x64_pd(_mm256_add_pd(wc, _mm256_permute4x64_pd(wc, _MM_PERM_DBCA)), _MM_PERM_DBCA);
    __m256d wb = _mm256_add_pd(wa, _mm256_permute4x64_pd(wa, _MM_PERM_DBCA));

    __m256d imm0 = _mm256_add_pd(_mm256_add_pd(z0, z1), _mm256_add_pd(z2, z3));
    __m256d imm1 = wb;

    return __m256dx2(imm0, imm1);
}

int main(){
    const __m256d a = _mm256_setr_pd(1, 2, 3, 4);
    const __m256d b = _mm256_setr_pd(5, 6, 7, 8);
    const __m256d c = _mm256_setr_pd(9, 10, 11, 12);
    const __m256d d = _mm256_setr_pd(13, 14, 15, 16);
    const __m256d e = _mm256_setr_pd(17, 18, 19, 20);
    
    __m256d x = _mm256_hadd_pd(a, a);
    __m256d y = _mm256_add_pd(a, _mm256_permute4x64_pd(a, _MM_PERM_CDAB));

    printf("end");
    getchar();
}
