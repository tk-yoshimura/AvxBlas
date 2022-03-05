#include "../avxblasutil.h"
#include <exception>

const int __mask_mm256[15] = { -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
const int __mask_mm128[7]  = { -1, -1, -1, 0, 0, 0, 0 };

__m256i mm256_mask(unsigned int n) {
#ifdef _DEBUG
    if (n >= 8) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm256_loadu_epi32(__mask_mm256 + (7 - (n)));
}

__m128i mm128_mask(unsigned int n) {
#ifdef _DEBUG
    if (n >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm_loadu_epi32(__mask_mm128 + (3 - (n)));
}