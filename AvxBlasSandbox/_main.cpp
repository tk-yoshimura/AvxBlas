#include <iostream>

#include <immintrin.h>

__forceinline double _mm_sum2to1_pd(const __m128d x) {
    const __m128d y = _mm_hadd_pd(x, x);
    const double ret = _mm_cvtsd_f64(y);

    return ret;
}

//__forceinline float _mm256_sum16to1_ps()

int main(){
    __m128d x1 = _mm_set_pd(2, 3);
    __m128d x2 = _mm_set_pd(11, 7);

    double y1 = _mm_sum2to1_pd(x1);
    double y2 = _mm_sum2to1_pd(x2);

    getchar();
}