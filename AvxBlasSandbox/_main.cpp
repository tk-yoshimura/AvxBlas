#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"
#include "../AvxBlas/Inline/inline_imcol_s.hpp"


int main(){
    const unsigned int n = 1, ic = 2, oc = 1, iw = 4, kw = 2, ow = iw - kw + 1;

    float* x = (float*)_aligned_malloc((n * ic * iw + 7) * sizeof(float), AVX2_ALIGNMENT);
    float* y = (float*)_aligned_malloc((n * oc * ow + 7) * sizeof(float), AVX2_ALIGNMENT);
    float* w = (float*)_aligned_malloc((ic * kw + 7) * sizeof(float), AVX2_ALIGNMENT);

    if (x == nullptr || y == nullptr || w == nullptr) {
        return -1;
    }

    x[0] = 1; x[1] = 2; x[2] = 3; x[3] = 4; x[4] = 5; x[5] = 6; x[6] = 7; x[7] = 8;
    w[0] = 8; w[1] = 7; w[2] = 6; w[3] = 5; w[4] = 4; w[5] = 3; w[6] = 2; w[7] = 1;

    imcol1d_padnone_unaligned_s(ic, oc, iw, 2, x, y, _mm256_setmask_ps(4));

    for (unsigned int i = 0; i < ic * kw; i++) {
        std::cout << y[i] << std::endl;
    }

    _aligned_free(x);
    _aligned_free(y);
    _aligned_free(w);

    getchar();
}
