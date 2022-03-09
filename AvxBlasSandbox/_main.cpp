#include <iostream>

#include <immintrin.h>

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1+e2+e3,e4+e5+e6+e7,-,-,-,-,-,-
__forceinline __m128 _mm256_hadd4_ps(const __m256 x) {
    const __m256i _perm82 = _mm256_setr_epi32(0, 4, 1, 2, 3, 5, 6, 7);

    const __m256 y = _mm256_hadd_ps(x, x);
    const __m256 z = _mm256_hadd_ps(y, y);
    const __m256 ret = _mm256_permutevar8x32_ps(z, _perm82);

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

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1,e2+e3,e4+e5,e6+e7,-,-,-,-
__forceinline __m256 _mm256_hadd2_ps(const __m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    const __m256 y = _mm256_hadd_ps(x, x);
    const __m256 ret = _mm256_permutevar8x32_ps(y, _perm);

    return ret;
}

int main(){
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

    __m256 x = _mm256_setr_ps(2, 3, 7, 11, 19, 23, 29, 37);
    
    __m256 y = _mm256_hadd4_ps(x);

    getchar();
}