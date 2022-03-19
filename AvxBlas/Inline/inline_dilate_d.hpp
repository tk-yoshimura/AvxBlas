#pragma once
#pragma unmanaged

#include "../utils.h"

// e0,e1,_,_ -> e0,e0,e1,e1
__forceinline __m256d _mm256_dilate2_pd(__m256d x) {
    const __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_BBAA);

    return ret;
}