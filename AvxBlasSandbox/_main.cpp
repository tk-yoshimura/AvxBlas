#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "../AvxBlas/Inline/inline_max_d.hpp"
#include "../AvxBlas/types.h"
#include "../AvxBlas/Util/util_mm_mask.cpp"

__forceinline __m256 _mm256_where_ps(__m256i cond, __m256 x, __m256 y) {
    __m256 ret = _mm256_blendv_ps(y, x, _mm256_castsi256_ps(cond));

    return ret;
}

int main(){
    const __m256i mask = _mm256_setmask_ps(4);

    const __m256 a = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256 b = _mm256_setr_ps(1, 0, 0, 0, 0, 0, 0, 0);
    const __m256 c = _mm256_setr_ps(NAN, INFINITY, -INFINITY, 0, NAN, INFINITY, -INFINITY, 0);
    
    const __m256 x = _mm256_where_ps(mask, a, _mm256_set1_ps(1));
    const __m256 y = _mm256_where_ps(mask, b, _mm256_set1_ps(1));
    const __m256 z = _mm256_where_ps(mask, c, _mm256_set1_ps(1));

    printf("end");
    getchar();
}
