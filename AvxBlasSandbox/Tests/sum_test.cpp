#include "../avxblas_sandbox.h"
#include "../../AvxBlas/Inline/inline_zeroset_s.hpp"
#include "../../AvxBlas/Inline/inline_copy_s.hpp"

int ag_sum_aligned_s_type1(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* s_ptr = (float*)_aligned_malloc((size_t)stride * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        zeroset_aligned_s(stride, s_ptr);

        for (uint j = 0; j < samples; j++) {
            float* sc_ptr = s_ptr;

            __m256 x0, x1, x2, x3;
            __m256 s0, s1, s2, s3;

            uint r = stride;

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_load_x4_ps(sc_ptr, s0, s1, s2, s3);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);
                s3 = _mm256_add_ps(x3, s3);

                _mm256_store_x4_ps(sc_ptr, s0, s1, s2, s3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                sc_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
            if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_load_x2_ps(x_ptr, x0, x1);
                _mm256_load_x2_ps(sc_ptr, s0, s1);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);

                _mm256_store_x2_ps(sc_ptr, s0, s1);

                x_ptr += AVX2_FLOAT_STRIDE * 2;
                sc_ptr += AVX2_FLOAT_STRIDE * 2;
                r -= AVX2_FLOAT_STRIDE * 2;
            }
            if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(x_ptr, x0);
                _mm256_load_x1_ps(sc_ptr, s0);

                s0 = _mm256_add_ps(x0, s0);

                _mm256_store_x1_ps(sc_ptr, s0);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
        }

        copy_aligned_s(stride, s_ptr, y_ptr);
        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int ag_sum_aligned_s_type2(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 zero = _mm256_setzero_ps();

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2, x3;
        __m256 s0, s1, s2, s3;

        uint r = stride;
        
        while (r >= AVX2_FLOAT_STRIDE * 4) {
            const float* xc_ptr = x_ptr;

            s0 = zero;  s1 = zero;  s2 = zero;  s3 = zero;

            for (uint j = 0; j < samples; j++) {
                _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                
                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);
                s3 = _mm256_add_ps(x3, s3);

                xc_ptr += stride;
            }

            _mm256_stream_x4_ps(y_ptr, s0, s1, s2, s3);

            x_ptr += AVX2_FLOAT_STRIDE * 4;
            y_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if(r >= AVX2_FLOAT_STRIDE * 3) {
            const float* xc_ptr = x_ptr;

            s0 = zero;  s1 = zero;  s2 = zero;

            for (uint j = 0; j < samples; j++) {
                _mm256_load_x3_ps(xc_ptr, x0, x1, x2);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);
                s2 = _mm256_add_ps(x2, s2);

                xc_ptr += stride;
            }

            _mm256_stream_x3_ps(y_ptr, s0, s1, s2);

            x_ptr += AVX2_FLOAT_STRIDE * 3;
            y_ptr += AVX2_FLOAT_STRIDE * 3;
            r -= AVX2_FLOAT_STRIDE * 3;
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            const float* xc_ptr = x_ptr;

            s0 = zero;  s1 = zero;

            for (uint j = 0; j < samples; j++) {
                _mm256_load_x2_ps(xc_ptr, x0, x1);

                s0 = _mm256_add_ps(x0, s0);
                s1 = _mm256_add_ps(x1, s1);

                xc_ptr += stride;
            }

            _mm256_stream_x2_ps(y_ptr, s0, s1);

            x_ptr += AVX2_FLOAT_STRIDE * 2;
            y_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        else if (r >= AVX2_FLOAT_STRIDE * 1) {
            const float* xc_ptr = x_ptr;

            s0 = zero;

            for (uint j = 0; j < samples; j++) {
                _mm256_load_x1_ps(xc_ptr, x0);

                s0 = _mm256_add_ps(x0, s0);

                xc_ptr += stride;
            }

            _mm256_stream_x1_ps(y_ptr, s0);

            x_ptr += AVX2_FLOAT_STRIDE;
            y_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }

        x_ptr += stride * (samples - 1);
    }

    return SUCCESS;
}