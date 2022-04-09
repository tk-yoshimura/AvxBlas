#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"
#include "../AvxBlas/Inline/inline_numeric.hpp"
#include "../AvxBlas/Inline/inline_loadstore_xn_s.hpp"
#include "../AvxBlas/Inline/inline_transpose_s.hpp"

__forceinline floatx2 float_linear1d(float xl, float xc, float xr) {
    float xc2 = xc + xc;
    float yl = xl + xc2;
    float yr = xr + xc2;

    return floatx2(yl, yr);
}

__forceinline __m256x2 _mm256_linear1d_ps(__m256 xl, __m256 xc, __m256 xr) {
    __m256 xc2 = _mm256_add_ps(xc, xc);
    __m256 yl = _mm256_add_ps(xl, xc2);
    __m256 yr = _mm256_add_ps(xr, xc2);

    return __m256x2(yl, yr);
}

int upsample1d_linear_c1(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (iw > 1u) {
        const __m256i srcmask = _mm256_setmask_ps((iw - 2u) & AVX2_FLOAT_REMAIN_MASK);
        const __m256i dstmask = _mm256_setmask_ps(((iw - 2u) * 2u) & AVX2_FLOAT_REMAIN_MASK);

        for (uint i = 0; i < n; i++) {
            {
                const float* xc_ptr = x_ptr;
                const float* xr_ptr = x_ptr + 1;

                float* yl_ptr = y_ptr;
                float* yr_ptr = yl_ptr + 1;

                float xc = *xc_ptr;
                float xr = *xr_ptr;

                floatx2 y = float_linear1d(xc, xc, xr);

                *yl_ptr = y.imm0;
                *yr_ptr = y.imm1;
            }
            {
                const float* xl_ptr = x_ptr + (iw - 2u);
                const float* xc_ptr = x_ptr + (iw - 1u);

                float* yl_ptr = y_ptr + (iw - 1u) * 2;
                float* yr_ptr = yl_ptr + 1;

                float xl = *xl_ptr;
                float xc = *xc_ptr;

                floatx2 y = float_linear1d(xl, xc, xc);

                *yl_ptr = y.imm0;
                *yr_ptr = y.imm1;
            }
            for (uint ix = 1, ox = 2; ix < iw - 1u; ix += AVX2_FLOAT_STRIDE, ox += AVX2_FLOAT_STRIDE * 2) {
                const uint ixl = ix - 1u, ixr = ix + 1u;

                const float* xl_ptr = x_ptr + ixl;
                const float* xc_ptr = x_ptr + ix;
                const float* xr_ptr = x_ptr + ixr;

                if (ix + AVX2_FLOAT_STRIDE <= iw - 1u) {
                    __m256 xl = _mm256_loadu_ps(xl_ptr);
                    __m256 xc = _mm256_loadu_ps(xc_ptr);
                    __m256 xr = _mm256_loadu_ps(xr_ptr);

                    __m256x2 y = _mm256_linear1d_ps(xl, xc, xr);
                    __m256x2 yt = _mm256_transpose8x2_ps(y.imm0, y.imm1);

                    _mm256_storeu_x2_ps(y_ptr + ox, yt.imm0, yt.imm1);
                }
                else {
                    __m256 xl = _mm256_maskload_ps(xl_ptr, srcmask);
                    __m256 xc = _mm256_maskload_ps(xc_ptr, srcmask);
                    __m256 xr = _mm256_maskload_ps(xr_ptr, srcmask);

                    __m256x2 y = _mm256_linear1d_ps(xl, xc, xr);
                    __m256x2 yt = _mm256_transpose8x2_ps(y.imm0, y.imm1);

                    if (ix + AVX2_FLOAT_STRIDE / 2 < iw - 1u) {
                        _mm256_maskstore_x2_ps(y_ptr + ox, yt.imm0, yt.imm1, dstmask);
                    }
                    else if (ix + AVX2_FLOAT_STRIDE / 2 == iw - 1u) {
                        _mm256_storeu_x1_ps(y_ptr + ox, yt.imm0);
                    }
                    else {
                        _mm256_maskstore_x1_ps(y_ptr + ox, yt.imm0, dstmask);
                    }

                    break;
                }
            }

            x_ptr += iw;
            y_ptr += ow;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            float x = *x_ptr;
            float y = x + x + x;

            y_ptr[0] = y_ptr[1] = y;

            x_ptr += 1;
            y_ptr += 2;
        }
    }

    return SUCCESS;
}

int main(){
    const unsigned int n = 5, c = 1;

    for (int iw = 1; iw <= 55; iw++) {
        int ow = iw * 2;

        float* x = (float*)_aligned_malloc((n * c * iw + 1) * sizeof(float), AVX2_ALIGNMENT);
        float* y = (float*)_aligned_malloc((n * c * ow + 1) * sizeof(float), AVX2_ALIGNMENT);

        if (x == nullptr || y == nullptr) {
            return -1;
        }

        for (int i = 0; i < (n * c * iw); i++) {
            x[i] = i;
        }

        upsample1d_linear_c1(n, c, iw, ow, x, y);

        for (int i = (n * c * ow - 1); i <= (n * c * ow); i++) {
            printf("y[%d] = %f\n", i, y[i]);
        }

        printf("\n");

        _aligned_free(x);
        _aligned_free(y);
    }

    printf("end");
    getchar();
}
