#include "../utils.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#pragma unmanaged

const int __mask_mm128[7] = { -1, -1, -1, 0, 0, 0, 0 };
const int __mask_mm256[15] = { -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };

__m128i _mm_setmask_ps(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm_loadu_epi32(__mask_mm128 + (3 - (n)));
}

__m256i _mm256_setmask_ps(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 8) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm256_loadu_epi32(__mask_mm256 + (7 - (n)));
}

__m128i _mm_setmask_pd(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 2) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm_loadu_epi32(__mask_mm128 + (3 - (n * 2)));
}

__m256i _mm256_setmask_pd(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm256_loadu_epi32(__mask_mm256 + (7 - (n * 2)));
}