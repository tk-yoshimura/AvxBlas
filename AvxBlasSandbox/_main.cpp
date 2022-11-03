#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "../AvxBlas/Inline/inline_max_d.hpp"
#include "../AvxBlas/types.h"

__forceinline __m256 _mm256_maxwise4_ps(__m256 x) {
    __m256 y = _mm256_max_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_max_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256 _mm256_softmaxexp_ps(__m256 x, __m256 x_max) {
    __m256 y = _mm256_sub_ps(x, x_max);
    __m256 z = _mm256_exp_ps(y);

    return z;
}

__forceinline __m256 _mm256_sumwise4_ps(__m256 x) {
    __m256 y = _mm256_add_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_add_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));

    return z;
}

int main(){
    __m256 x0 = _mm256_setr_ps(9, 0, 4, -6, 0, 0, 0, 0);

    __m256 x0_max = _mm256_maxwise4_ps(x0);

    __m256 y0 = _mm256_softmaxexp_ps(x0, x0_max);

    __m256 y0_sum = _mm256_sumwise4_ps(y0);

    __m256 z0 = _mm256_div_ps(y0, y0_sum);
    
    printf("end");
    getchar();
}
