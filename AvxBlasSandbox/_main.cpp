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


int main(){
    const __m256i mask = _mm256_setmask_ps(1);
    const __m256 minf = _mm256_set1_ps(-HUGE_VALF);

    __m256 x0 = _mm256_setr_ps(12, 2, -15, 0, 13, -13, 11, -3);
    __m256 x1 = _mm256_setr_ps(-7, NAN, NAN, NAN, NAN, NAN, NAN, NAN);

    x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), minf));

    __m256 x01_max = _mm256_max_ps(_mm256_maxwise8_ps(x0), _mm256_maxwise8_ps(x1));

    __m256 y0 = _mm256_softmaxexp_ps(x0, x01_max);
    __m256 y1 = _mm256_softmaxexp_ps(x1, x01_max);

    __m256 y01_sum = _mm256_add_ps(_mm256_sumwise8_ps(y0), _mm256_sumwise8_ps(y1));

    __m256 z0 = _mm256_div_ps(y0, y01_sum);
    __m256 z1 = _mm256_div_ps(y1, y01_sum);
    
    printf("end");
    getchar();
}
