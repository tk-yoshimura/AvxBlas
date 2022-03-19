#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"

// (s, c) += a * b
__forceinline void _mm256_fasttwosum_fmadd_ps(const __m256 a, const __m256 b, __m256& s, __m256& c) {
    __m256 tmp = s;
    s = _mm256_fmadd_ps(a, b, s);
    c = _mm256_add_ps(c, _mm256_fmadd_ps(a, b, _mm256_sub_ps(tmp, s)));
}

// (s, c) += a * b
__forceinline void _mm256_twosum_fmadd_ps(const __m256 a, const __m256 b, __m256& s, __m256& c) {
    __m256 tmp = s;
    s = _mm256_fmadd_ps(a, b, _mm256_add_ps(c, s));
    c = _mm256_add_ps(c, _mm256_fmadd_ps(a, b, _mm256_sub_ps(tmp, s)));
}

int main(){
    const int N = 10000000;

    srand((unsigned int)time(NULL));

    {
        const float u = 0.1234, v = 123;
        const double expect = (double)u * (double)v * N;

        __m256 a = _mm256_set1_ps(0.1234);
        __m256 b = _mm256_set1_ps(123);

        __m256 s = _mm256_setzero_ps(), c = _mm256_setzero_ps();
        __m256 x = _mm256_setzero_ps();

        for (int i = 0; i < N; i++) {
            x = _mm256_fmadd_ps(a, b, x);
            _mm256_twosum_fmadd_ps(a, b, s, c);
        }

        float actual_fma = _mm256_cvtss_f32(x);
        float actual_twosum = _mm256_cvtss_f32(s);

        printf("%.15e\n", expect);
        printf("%.7e\n", actual_twosum);
        printf("%.7e\n\n", actual_fma);
    }

    for (int j = 0; j < 32; j++) {
        __m256 s = _mm256_setzero_ps(), c = _mm256_setzero_ps();
        __m256 x = _mm256_setzero_ps();

        double expect = 0;

        for (int i = 0; i < N; i++) {
            float u = ((rand() % 10001) - 5000) * 0.0001f;
            float v = (rand() % 101) - 50;

            __m256 a = _mm256_set1_ps(u);
            __m256 b = _mm256_set1_ps(v);

            x = _mm256_fmadd_ps(a, b, x);
            _mm256_twosum_fmadd_ps(a, b, s, c);

            expect += (double)u * (double)v;
        }

        float actual_fma = _mm256_cvtss_f32(x);
        float actual_twosum = _mm256_cvtss_f32(s);

        printf("%.15e\n", expect);
        printf("%.7e\n", actual_twosum);
        printf("%.7e\n\n", actual_fma);
    }

    getchar();
}
