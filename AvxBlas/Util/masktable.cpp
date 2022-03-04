#include "../AvxBlasUtil.h"
#include <exception>

__m256i AvxBlas::masktable_m256(unsigned int k) {
#ifdef _DEBUG
    if (k >= 8) {
        throw std::exception();
    }
#endif // _DEBUG

    int v[15] = { -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
    int j = 7 - k;
    
    return _mm256_setr_epi32(v[j], v[j + 1], v[j + 2], v[j + 3], v[j + 4], v[j + 5], v[j + 6], v[j + 7]);
}

__m128i AvxBlas::masktable_m128(unsigned int k) {
#ifdef _DEBUG
    if (k >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    int v[7] = { -1, -1, -1, 0, 0, 0, 0 };
    int j = 3 - k;

    return _mm_setr_epi32(v[j], v[j + 1], v[j + 2], v[j + 3]);
}