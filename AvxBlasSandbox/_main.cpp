#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"
#include "../AvxBlas/Inline/inline_numeric.hpp"
#include "../AvxBlas/Inline/inline_loadstore_xn_s.hpp"
#include "../AvxBlas/Inline/inline_transpose_s.hpp"

__forceinline __m128 _mm_evensort_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmp_ps(x, y, _CMP_GT_OQ), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

__forceinline __m128 _mm_oddsort_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmp_ps(x, y, _CMP_GT_OQ), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

__forceinline __m128 _mm_sort_ps(__m128 x) {
    x = _mm_oddsort_ps(x);
    x = _mm_evensort_ps(x);
    x = _mm_oddsort_ps(x);

    return x;
}

int main(){
    __m128 x1234 = _mm_setr_ps(1, 2, 3, 4);
    __m128 x1243 = _mm_setr_ps(1, 2, 4, 3);
    __m128 x1324 = _mm_setr_ps(1, 3, 2, 4);
    __m128 x1342 = _mm_setr_ps(1, 3, 4, 2);
    __m128 x1423 = _mm_setr_ps(1, 4, 2, 3);
    __m128 x1432 = _mm_setr_ps(1, 4, 3, 2);

    __m128 x2134 = _mm_setr_ps(2, 1, 3, 4);
    __m128 x2143 = _mm_setr_ps(2, 1, 4, 3);
    __m128 x2341 = _mm_setr_ps(2, 3, 4, 1);
    __m128 x2431 = _mm_setr_ps(2, 4, 3, 1);
    __m128 x2314 = _mm_setr_ps(2, 3, 1, 4);
    __m128 x2413 = _mm_setr_ps(2, 4, 1, 3);

    __m128 x3124 = _mm_setr_ps(3, 1, 2, 4);
    __m128 x3142 = _mm_setr_ps(3, 1, 4, 2);
    __m128 x3214 = _mm_setr_ps(3, 2, 1, 4);
    __m128 x3412 = _mm_setr_ps(3, 4, 1, 2);
    __m128 x3241 = _mm_setr_ps(3, 2, 4, 1);
    __m128 x3421 = _mm_setr_ps(3, 4, 2, 1);

    __m128 x4123 = _mm_setr_ps(4, 1, 2, 3);
    __m128 x4132 = _mm_setr_ps(4, 1, 3, 2);
    __m128 x4213 = _mm_setr_ps(4, 2, 1, 3);
    __m128 x4312 = _mm_setr_ps(4, 3, 1, 2);
    __m128 x4231 = _mm_setr_ps(4, 2, 3, 1);
    __m128 x4321 = _mm_setr_ps(4, 3, 2, 1);
               
    __m128 y1234 = _mm_sort_ps(x1234);
    __m128 y1243 = _mm_sort_ps(x1243);
    __m128 y1324 = _mm_sort_ps(x1324);
    __m128 y1342 = _mm_sort_ps(x1342);
    __m128 y1423 = _mm_sort_ps(x1423);
    __m128 y1432 = _mm_sort_ps(x1432);

    __m128 y2134 = _mm_sort_ps(x2134);
    __m128 y2143 = _mm_sort_ps(x2143);
    __m128 y2341 = _mm_sort_ps(x2341);
    __m128 y2431 = _mm_sort_ps(x2431);
    __m128 y2314 = _mm_sort_ps(x2314);
    __m128 y2413 = _mm_sort_ps(x2413);

    __m128 y3124 = _mm_sort_ps(x3124);
    __m128 y3142 = _mm_sort_ps(x3142);
    __m128 y3214 = _mm_sort_ps(x3214);
    __m128 y3412 = _mm_sort_ps(x3412);
    __m128 y3241 = _mm_sort_ps(x3241);
    __m128 y3421 = _mm_sort_ps(x3421);

    __m128 y4123 = _mm_sort_ps(x4123);
    __m128 y4132 = _mm_sort_ps(x4132);
    __m128 y4213 = _mm_sort_ps(x4213);
    __m128 y4312 = _mm_sort_ps(x4312);
    __m128 y4231 = _mm_sort_ps(x4231);
    __m128 y4321 = _mm_sort_ps(x4321);

    printf("end");
    getchar();
}
