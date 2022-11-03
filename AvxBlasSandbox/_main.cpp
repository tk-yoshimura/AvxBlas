#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "../AvxBlas/Inline/inline_max_d.hpp"
#include "../AvxBlas/types.h"
#include "../AvxBlas/Util/util_mm_mask.cpp"

__forceinline __m256 _mm256_softmaxexp_ps(__m256 x, __m256 x_max) {
    __m256 y = _mm256_sub_ps(x, x_max);
    __m256 z = _mm256_exp_ps(y);

    return z;
}

__forceinline __m256 _mm256_maxwise4_ps(__m256 x) {
    __m256 y = _mm256_max_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_max_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256 _mm256_sumwise4_ps(__m256 x) {
    __m256 y = _mm256_add_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_add_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256 _mm256_maxwise8_ps(__m256 x) {
    __m256 y = _mm256_max_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_max_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));
    __m256 w = _mm256_max_ps(z, _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(z), _MM_PERM_BADC)));

    return w;
}

__forceinline __m256 _mm256_sumwise8_ps(__m256 x) {
    __m256 y = _mm256_add_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_add_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));
    __m256 w = _mm256_add_ps(z, _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(z), _MM_PERM_BADC)));

    return w;
}

__forceinline __m256 _mm256_normal_asone_ps(__m256 x) {    
    __m256 y = _mm256_and_ps(_mm256_set1_ps(NAN), _mm256_cmp_ps(x, x, _CMP_NEQ_UQ));
    __m256 z = _mm256_add_ps(_mm256_set1_ps(1), y);

    return z;
}

int main(){
    const __m256 fills = _mm256_set1_ps(1);

    __m256 x = _mm256_setr_ps(-7, -HUGE_VALF, NAN, NAN, HUGE_VALF, NAN, 0, NAN);
    __m256 y = _mm256_normal_asone_ps(x);

    printf("end");
    getchar();
}
