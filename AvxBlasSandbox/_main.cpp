#include <iostream>

#include <immintrin.h>

__forceinline float _mm_sum4to1_ps(const __m128 x) {
    const __m128 y = _mm_add_ps(x, _mm_movehl_ps(x, x));
    const float ret = _mm_cvtss_f32(_mm_add_ss(y, _mm_shuffle_ps(y, y, 1)));

    return ret;
}

int main(){
    __m128 x1 = _mm_set_ps(2, 3, 7, 11);
    __m128 x2 = _mm_set_ps(11, 7, 19, 23);

    float y1 = _mm_sum4to1_ps(x1);
    float y2 = _mm_sum4to1_ps(x2);

    getchar();
}