#include "../utils.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#pragma unmanaged

#define _MASK(i, n) (((i) < (n)) ? ~0u : 0u)

__m128i _mm_setmask_ps(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm_setr_epi32(_MASK(0, n), _MASK(1, n), _MASK(2, n), _MASK(3, n));
}

__m256i _mm256_setmask_ps(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 8) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm256_setr_epi32(_MASK(0, n), _MASK(1, n), _MASK(2, n), _MASK(3, n), _MASK(4, n), _MASK(5, n), _MASK(6, n), _MASK(7, n));
}

__m128i _mm_setmask_pd(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 2) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm_setr_epi32(_MASK(0, n), _MASK(0, n), _MASK(1, n), _MASK(1, n));
}

__m256i _mm256_setmask_pd(const unsigned int n) {
#ifdef _DEBUG
    if (n >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm256_setr_epi32(_MASK(0, n), _MASK(0, n), _MASK(1, n), _MASK(1, n), _MASK(2, n), _MASK(2, n), _MASK(3, n), _MASK(3, n));
}