#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "../AvxBlas/Inline/inline_max_d.hpp"
#include "../AvxBlas/types.h"
#include "../AvxBlas/Util/util_mm_mask.cpp"

__forceinline __m256i _mm256_cvtepi64x2_epi32(__m256i a, __m256i b) {
    __m256i y = _mm256_castpd_si256(
        _mm256_permute4x64_pd(
            _mm256_castps_pd(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), 0b10001000)), 
            _MM_PERM_DBCA)
    );

    return y;
}

int main(){
    const __m256i a = _mm256_setr_epi64x(-1L, 255L, -65535L, 16777215L);
    const __m256i b = _mm256_setr_epi64x(-33L, 37L, -53L, 59L);
    const __m256i c = _mm256_setr_epi64x(-2147483648L, 2147483647L, 65535L, -16777215L);
    const __m256i d = _mm256_setr_epi64x(33000000L, -370000000L, 53000000L, -590000000L);
    const __m256i e = _mm256_setr_epi64x(1L, -16L, 256L, -1024L);
    const __m256i f = _mm256_setr_epi64x(4096L, -16384L, 65536L, 0);

    __m256i x = _mm256_cvtepi64x2_epi32(a, b);
    __m256i y = _mm256_cvtepi64x2_epi32(c, d);
    __m256i z = _mm256_cvtepi64x2_epi32(e, f);

    printf("end");
    getchar();
}
