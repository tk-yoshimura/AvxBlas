#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"
#include "../AvxBlas/Inline/inline_numeric.hpp"
#include "../AvxBlas/Inline/inline_loadstore_xn_s.hpp"
#include "../AvxBlas/Inline/inline_transpose_s.hpp"

void sort_s(uint n, float* v_ptr) {
    for (uint h = n * 3 / 4; h >= AVX2_FLOAT_STRIDE; h = h * 3 / 4) {
        for (uint i = 0; i + h + AVX2_FLOAT_STRIDE <= n; i += AVX2_FLOAT_STRIDE / 2) {
            __m256 a = _mm256_loadu_ps(v_ptr + i);
            __m256 b = _mm256_loadu_ps(v_ptr + i + h);

            __m256 x = _mm256_min_ps(a, b);
            __m256 y = _mm256_max_ps(a, b);

            _mm256_storeu_ps(v_ptr + i, x);
            _mm256_storeu_ps(v_ptr + i + h, y);
        }
        {
            uint i = n - h - AVX2_FLOAT_STRIDE;

            __m256 a = _mm256_loadu_ps(v_ptr + i);
            __m256 b = _mm256_loadu_ps(v_ptr + i + h);

            __m256 x = _mm256_min_ps(a, b);
            __m256 y = _mm256_max_ps(a, b);

            _mm256_storeu_ps(v_ptr + i, x);
            _mm256_storeu_ps(v_ptr + i + h, y);
        }
    }
    if (n >= AVX1_FLOAT_STRIDE * 2) {
        uint h = AVX1_FLOAT_STRIDE;

        for (uint i = 0; i + h + AVX1_FLOAT_STRIDE <= n; i += AVX1_FLOAT_STRIDE / 2) {
            __m128 a = _mm_loadu_ps(v_ptr + i);
            __m128 b = _mm_loadu_ps(v_ptr + i + h);

            __m128 x = _mm_min_ps(a, b);
            __m128 y = _mm_max_ps(a, b);

            _mm_storeu_ps(v_ptr + i, x);
            _mm_storeu_ps(v_ptr + i + h, y);
        }
        {
            uint i = n - h - AVX1_FLOAT_STRIDE;

            __m128 a = _mm_loadu_ps(v_ptr + i);
            __m128 b = _mm_loadu_ps(v_ptr + i + h);

            __m128 x = _mm_min_ps(a, b);
            __m128 y = _mm_max_ps(a, b);

            _mm_storeu_ps(v_ptr + i, x);
            _mm_storeu_ps(v_ptr + i + h, y);
        }
    }
}

__forceinline __m256 _mm256_isnan_ps(__m256 x) {
    __m256 y = _mm256_cmp_ps(x, x, _CMP_NEQ_UQ);

    return y;
}

__forceinline __m256 _mm256_not_ps(__m256 x) {
    const __m256 setbits = _mm256_castsi256_ps(_mm256_set1_epi32(~0u));
 
    __m256 y = _mm256_xor_ps(x, setbits);

    return y;
}

__forceinline __m256 _mm256_cmpgt_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    
    return gt;
}

__forceinline __m256 _mm256_cmpnangt_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    __m256 xisnan = _mm256_isnan_ps(x);
    __m256 yisnan = _mm256_isnan_ps(y);
    __m256 bothnan = _mm256_and_ps(xisnan, yisnan);

    __m256 ret = _mm256_andnot_ps(bothnan, _mm256_or_ps(gt, yisnan));

    return ret;
}

__forceinline void _mm256_cmpnangtswap_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpnangt_ps(a, b);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));
}

__forceinline bool _mm256_cmpnangtswap_signal_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpnangt_ps(a, b);
    
    bool swaped = _mm256_movemask_ps(gtflag) > 0;
    
    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return swaped;
}

__forceinline bool _mm256_cmpnangtswap_masksignal_ps(__m256 a, __m256 b, __m256i mask, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_and_ps(_mm256_cmpnangt_ps(a, b), _mm256_castsi256_ps(mask));

    bool swaped = _mm256_movemask_ps(gtflag) > 0;

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return swaped;
}

int main(){
    //const uint N = 255;
    //
    //srand((uint)time(NULL));
    //
    //float* v = (float*)_aligned_malloc((N + 1) * sizeof(float), AVX2_ALIGNMENT);
    //
    //if (v == nullptr) {
    //    return -1;
    //}
    //
    //for (uint i = 0; i < N; i++) {
    //    v[i] = (rand() % 1001) * 0.001f;
    //}
    //
    //v[N] = -1;
    //
    //sort_s(N, v);
    //
    //for (uint i = 0; i < N; i++) {
    //    printf("%.3lf\n", v[i]);
    //}

    //__m256 a = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
    //__m256 b = _mm256_setr_ps(6, 5, 7, 2, 4, 3, 8, 9);
    //__m256 c = _mm256_setr_ps(6, 5, 7, 2, 4, 3, 8, NAN);
    //__m256 x, y;
    //
    //bool swaped = _mm256_cmpgtswap_signal_ps(a, b, x, y);
    //
    //swaped = _mm256_cmpgtswap_signal_ps(b, a, x, y);
    //
    //swaped = _mm256_cmpgtswap_signal_ps(a, a, x, y);
    //
    //swaped = _mm256_cmpgtswap_signal_ps(c, b, x, y);
    //
    //swaped = _mm256_cmpgtswap_signal_ps(b, c, x, y);
    //
    //swaped = _mm256_cmpgtswap_signal_ps(c, c, x, y);

    __m256 a = _mm256_setr_ps(1, 1, 2, NAN, 1,   NAN, INFINITY, INFINITY);
    __m256 b = _mm256_setr_ps(1, 2, 1, 1,   NAN, NAN, INFINITY, NAN     );


    __m256 c = _mm256_cmpnangt_ps(a, b);


    printf("end");
    getchar();
}
